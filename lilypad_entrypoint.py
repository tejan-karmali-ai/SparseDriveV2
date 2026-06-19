"""Lilypad entrypoints for SparseDriveV2 training and evaluation."""

import logging
import os
import subprocess
import sys
from typing import Any

import ray

from entrypoint_common import (
    download_hf_checkpoint,
    download_s3_archive,
    download_s3_prefix,
    setup_environment,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# HuggingFace URLs for pretrained checkpoints
HF_CHECKPOINT_URLS = {
    "v1": "https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main/sparsedrive_navsimv1_92p2.ckpt",
    "v2": "https://huggingface.co/wenchaosun/SparseDriveV2/resolve/main/sparsedrive_navsimv2_90p3.ckpt",
}

TRAINING_SCRIPT = "navsim/planning/script/run_training.py"

EVAL_SCRIPTS = {
    "v2": "navsim/planning/script/run_pdm_score_navtest_v2_fast.py",
    "v1": "navsim/planning/script/run_pdm_score_navtest_v1_fast.py",
}
METRIC_CACHING_SCRIPTS = {
    "v2": "navsim/planning/script/run_metric_caching.py",
    "v1": "navsim/planning/script/run_metric_caching_v1.py",
}

# Where to put dataset and cache on the Lilypad worker
DATASET_LOCAL = "/tmp/navsim_data"      # OPENSCENE_DATA_ROOT
CACHE_LOCAL = "/tmp/training_cache"     # cache_path Hydra override

METRIC_CACHE_LOCAL = "/tmp/metric_cache"
TEST_CACHE_LOCAL = "/tmp/test_cache"
NAVSIM_LOGS_LOCAL = os.path.join(DATASET_LOCAL, "navsim_logs", "test")

# OCI S3 endpoints per region
OCI_S3_ENDPOINTS = {
    "us-chicago-1": "https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com",
    "us-phoenix-1": "https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com",
}

# research-datasets bucket lives in OCI Phoenix
RESEARCH_DATASETS_ENDPOINT = OCI_S3_ENDPOINTS["us-phoenix-1"]
RESEARCH_DATASETS_REGION = "us-phoenix-1"


def entrypoint_fn(config: dict[Any, Any]) -> None:
    """
    Main Lilypad entrypoint for SparseDriveV2 training.

    Config keys:
        num_gpus (int): Number of GPUs to use (1, 4, or 8). Default: 8.
        dataset_version (str): "v1" or "v2". Default: "v2".

        -- Data (choose one) --
        training_cache_s3_path (str|None): S3 prefix containing a pre-computed
            training cache. If set, downloads and uses use_cache_without_dataset=True.
        dataset_s3_path (str|None): S3 prefix of raw NAVSIM data
            (e.g. "s3://tejan-experiments/navsim"). If set, downloads and
            computes the cache on the worker.

        -- Optional: save computed cache back to S3 ---
        cache_s3_output (str|None): After training, upload the computed cache
            to this S3 prefix so future runs can skip recomputation.

        -- Checkpoint --
        pretrained_checkpoint (str|None): "v1", "v2", or a full URL. Downloads
            from HuggingFace before training.

        -- Training hyperparameters --
        experiment_name (str)
        max_epochs (int)
        batch_size (int)
        lr (float)
        num_workers (int)
        extra_overrides (list[str]): Additional raw Hydra override strings.

        -- W&B --
        wandb.enable (bool)
    """
    num_gpus = config.get("num_gpus", 8)
    logger.info("Starting SparseDriveV2 training with %d GPU(s)", num_gpus)

    setup_environment(install_requirements=True, install_local_pkg=True)

    if not ray.is_initialized():
        ray.init()

    if num_gpus <= 1:
        _run_single_gpu(config)
    else:
        _run_multi_gpu(config, num_gpus)


def _run_single_gpu(config):
    @ray.remote(num_gpus=1)
    def _train(cfg):
        _run_training(cfg, num_gpus=1)

    ray.get(_train.remote(config))


def _run_multi_gpu(config, num_gpus):
    @ray.remote(num_gpus=num_gpus)
    def _train(cfg, n_gpus):
        _run_training(cfg, num_gpus=n_gpus)

    ray.get(_train.remote(config, num_gpus))


def _run_training(config, num_gpus):
    """Download assets, set env vars, and launch training inside a Ray GPU task."""
    from entrypoint_common import (
        download_hf_checkpoint,
        download_s3_archive,
        download_s3_prefix,
    )

    repo_root = os.path.abspath(os.path.dirname(__file__))

    # --- env vars required by navsim Hydra config ---
    os.environ["NAVSIM_DEVKIT_ROOT"] = repo_root
    os.environ["NAVSIM_EXP_ROOT"] = "/tmp/navsim_exp"
    os.environ["OPENSCENE_DATA_ROOT"] = DATASET_LOCAL

    # --- decide data mode ---
    cache_s3 = config.get("training_cache_s3_path")
    dataset_s3 = config.get("dataset_s3_path")

    if cache_s3:
        # Fast path: pre-computed cache already on S3
        logger.info("Downloading pre-computed training cache from %s ...", cache_s3)
        if cache_s3.endswith((".tar", ".tar.gz")):
            download_s3_archive(cache_s3, CACHE_LOCAL)
        else:
            download_s3_prefix(cache_s3, CACHE_LOCAL)
        use_cache = True

    elif dataset_s3:
        # Slow path: download raw NAVSIM data, compute cache during training.
        # research-datasets bucket is in OCI Phoenix — use its endpoint regardless
        # of which region the cluster job runs in.
        logger.info("Downloading raw NAVSIM dataset from %s (OCI Phoenix) ...", dataset_s3)
        _download_navsim_dataset(
            dataset_s3,
            DATASET_LOCAL,
            endpoint_url=RESEARCH_DATASETS_ENDPOINT,
            region=RESEARCH_DATASETS_REGION,
        )
        use_cache = False

    else:
        raise ValueError(
            "Set either 'training_cache_s3_path' (fast) or 'dataset_s3_path' (compute cache)."
        )

    # --- optional: download pretrained checkpoint from HuggingFace ---
    ckpt_path = None
    pretrained = config.get("pretrained_checkpoint")
    if pretrained:
        url = HF_CHECKPOINT_URLS.get(pretrained, pretrained)
        filename = url.split("/")[-1]
        ckpt_path = f"/tmp/checkpoints/{filename}"
        download_hf_checkpoint(url, ckpt_path)

    # --- build Hydra overrides ---
    dataset_version = config.get("dataset_version", "v2")
    experiment_name = config.get(
        "experiment_name", f"sparsedrive_navsimv{dataset_version}_lilypad"
    )
    max_epochs = config.get("max_epochs", 10)
    batch_size = config.get("batch_size", 16)
    lr = config.get("lr", 1e-4)
    num_workers = config.get("num_workers", 16)

    overrides = [
        "--config-name", "default_training",
        "agent=sparsedrive_agent",
        f"experiment_name={experiment_name}",
        "train_test_split=navtrain",
        f"trainer.params.max_epochs={max_epochs}",
        f"trainer.params.devices={num_gpus}",
        "trainer.params.num_nodes=1",
        f"dataloader.params.batch_size={batch_size}",
        f"dataloader.params.num_workers={num_workers}",
        "dataloader.params.prefetch_factor=4",
        f"agent.lr={lr}",
    ]

    if use_cache:
        overrides += [
            "use_cache_without_dataset=True",
            "force_cache_computation=False",
            f"cache_path={CACHE_LOCAL}",
        ]
    else:
        overrides += [
            "use_cache_without_dataset=False",
            "force_cache_computation=True",
            f"cache_path={CACHE_LOCAL}",
        ]

    if ckpt_path:
        overrides.append(f"agent.checkpoint_path={ckpt_path}")

    if dataset_version == "v1":
        overrides += [
            "+agent.config.dataset_version=v1",
            "+agent.config.velocity_filter_num=[64,20]",
        ]

    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enable", True):
        os.environ.setdefault("WANDB_RUN_NAME", experiment_name)

    overrides += config.get("extra_overrides", [])

    # --- build command ---
    script = os.path.join(repo_root, TRAINING_SCRIPT)
    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            script,
        ] + overrides
    else:
        cmd = [sys.executable, script] + overrides

    logger.info("Running: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ,
    )
    for line in proc.stdout:
        print(f"[train] {line.rstrip()}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # --- optional: save computed cache to S3 for future reuse ---
    cache_s3_output = config.get("cache_s3_output")
    if cache_s3_output and not use_cache and os.path.isdir(CACHE_LOCAL):
        logger.info("Uploading computed cache to %s ...", cache_s3_output)
        _upload_cache(CACHE_LOCAL, cache_s3_output)


def _download_navsim_dataset(dataset_s3, local_dir, endpoint_url=None, region=None):
    """
    Download maps, navsim_logs/trainval, and sensor_blobs/trainval from S3.
    Expected S3 structure (from download_navsim_lilypad.py):
        <prefix>/maps/
        <prefix>/navsim_logs/trainval/
        <prefix>/sensor_blobs/trainval/
    """
    os.makedirs(local_dir, exist_ok=True)
    prefix = dataset_s3.rstrip("/")
    # Download in order of size: maps (small) → logs (small) → sensor blobs (large)
    for sub in ("maps", "navsim_logs/trainval", "sensor_blobs/trainval"):
        logger.info("Downloading %s/%s ...", prefix, sub)
        download_s3_prefix(
            f"{prefix}/{sub}",
            os.path.join(local_dir, sub),
            endpoint_url=endpoint_url,
            region=region,
        )


def _upload_cache(local_dir, s3_prefix):
    """Upload local cache directory to S3."""
    import boto3
    import botocore.config

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-chicago-1")
    config = botocore.config.Config(
        connect_timeout=30,
        read_timeout=300,
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    s3 = boto3.client("s3", endpoint_url=endpoint, region_name=region, config=config)

    path = s3_prefix[5:]  # strip "s3://"
    bucket, key_prefix = path.split("/", 1)

    count = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            rel = os.path.relpath(full_path, local_dir)
            key = f"{key_prefix.rstrip('/')}/{rel}"
            s3.upload_file(full_path, bucket, key)
            count += 1
    logger.info("Uploaded %d cache files to %s", count, s3_prefix)


# ---------------------------------------------------------------------------
# Evaluation entrypoint
# ---------------------------------------------------------------------------

def eval_entrypoint_fn(config: dict[Any, Any]) -> None:
    """
    Lilypad entrypoint for SparseDriveV2 evaluation (PDM scoring on navtest).

    Config keys:
        num_gpus (int): GPUs for model inference (1 or 4). Default: 1.
        dataset_version (str): "v1" or "v2". Default: "v2".
        pretrained_checkpoint (str): "v1", "v2", or full URL.

        -- Precomputed caches (fast path) --
        test_cache_s3_path (str|None): S3 prefix of precomputed test feature cache.
        metric_cache_s3_path (str|None): S3 prefix of precomputed PDM metric cache.

        -- Raw data (compute caches on-cluster) --
        dataset_s3_path (str|None): S3 prefix of raw NAVSIM data. Used to compute
            the test feature cache and metric cache if the above are null.
            Downloads navsim_logs/test + sensor_blobs/test. Uses Phoenix endpoint.
        test_cache_s3_output (str|None): Upload computed test cache here for reuse.
        metric_cache_s3_output (str|None): Upload computed metric cache here for reuse.

        -- Output --
        results_s3_output (str): S3 prefix to upload the results CSV.
        experiment_name (str)
        batch_size (int): Inference batch size.
        extra_overrides (list[str]): Additional Hydra override strings.
    """
    num_gpus = config.get("num_gpus", 1)
    logger.info("Starting SparseDriveV2 evaluation with %d GPU(s)", num_gpus)

    setup_environment(install_requirements=True, install_local_pkg=True)

    if not ray.is_initialized():
        ray.init()

    if num_gpus <= 1:
        @ray.remote(num_gpus=1)
        def _eval(cfg):
            _run_evaluation(cfg, num_gpus=1)
        ray.get(_eval.remote(config))
    else:
        @ray.remote(num_gpus=num_gpus)
        def _eval(cfg, n_gpus):
            _run_evaluation(cfg, num_gpus=n_gpus)
        ray.get(_eval.remote(config, num_gpus))


def _run_evaluation(config, num_gpus):
    """Download assets and run PDM scoring inside a Ray GPU task."""
    repo_root = os.path.abspath(os.path.dirname(__file__))

    os.environ["NAVSIM_DEVKIT_ROOT"] = repo_root
    os.environ["NAVSIM_EXP_ROOT"] = "/tmp/navsim_exp"
    os.environ["OPENSCENE_DATA_ROOT"] = DATASET_LOCAL

    dataset_version = config.get("dataset_version", "v2")

    # --- pretrained checkpoint ---
    pretrained = config.get("pretrained_checkpoint", "v2")
    url = HF_CHECKPOINT_URLS.get(pretrained, pretrained)
    filename = url.split("/")[-1]
    ckpt_path = f"/tmp/checkpoints/{filename}"
    download_hf_checkpoint(url, ckpt_path)

    # --- test feature cache ---
    test_cache_s3 = config.get("test_cache_s3_path")
    if test_cache_s3:
        logger.info("Downloading test feature cache from %s ...", test_cache_s3)
        download_s3_prefix(test_cache_s3, TEST_CACHE_LOCAL)
    elif config.get("dataset_s3_path"):
        _compute_test_cache(config, repo_root, dataset_version)
    else:
        raise ValueError("Set 'test_cache_s3_path' or 'dataset_s3_path' to provide test features.")

    # --- metric cache ---
    metric_cache_s3 = config.get("metric_cache_s3_path")
    if metric_cache_s3:
        logger.info("Downloading metric cache from %s ...", metric_cache_s3)
        download_s3_prefix(metric_cache_s3, METRIC_CACHE_LOCAL)
    elif config.get("dataset_s3_path"):
        _compute_metric_cache(config, repo_root, dataset_version)
    else:
        raise ValueError("Set 'metric_cache_s3_path' or 'dataset_s3_path' to provide metric cache.")

    # --- navsim_logs/test is also needed by SceneLoader for token discovery ---
    # If we only have caches (no dataset_s3_path), download just the logs (small).
    logs_local = os.path.join(DATASET_LOCAL, "navsim_logs", "test")
    if not os.path.isdir(logs_local):
        navsim_logs_s3 = config.get("navsim_logs_s3_path") or (
            config.get("dataset_s3_path", "").rstrip("/") + "/navsim_logs/test"
            if config.get("dataset_s3_path") else None
        )
        if navsim_logs_s3:
            logger.info("Downloading navsim_logs/test from %s ...", navsim_logs_s3)
            download_s3_prefix(
                navsim_logs_s3, logs_local,
                endpoint_url=RESEARCH_DATASETS_ENDPOINT,
                region=RESEARCH_DATASETS_REGION,
            )

    experiment_name = config.get("experiment_name", f"sparsedrive_navsimv{dataset_version}_eval")
    batch_size = config.get("batch_size", 8)

    overrides = [
        f"agent=sparsedrive_agent",
        f"agent.checkpoint_path={ckpt_path}",
        f"experiment_name={experiment_name}",
        f"metric_cache_path={METRIC_CACHE_LOCAL}",
        f"+test_cache_path={TEST_CACHE_LOCAL}",
        f"dataloader.params.batch_size={batch_size}",
        f"trainer.params.devices={num_gpus}",
        "trainer.params.num_nodes=1",
    ]

    if dataset_version == "v1":
        overrides += [
            "+agent.config.dataset_version=v1",
            "+agent.config.velocity_filter_num=[64,20]",
            "+agent.config.metrics=[no_at_fault_collisions,drivable_area_compliance,driving_direction_compliance,time_to_collision_within_bound,comfort,ego_progress]",
        ]

    overrides += config.get("extra_overrides", [])

    script = os.path.join(repo_root, EVAL_SCRIPTS[dataset_version])
    if num_gpus > 1:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={num_gpus}", script] + overrides
    else:
        cmd = [sys.executable, script] + overrides

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=os.environ,
    )
    for line in proc.stdout:
        print(f"[eval] {line.rstrip()}", flush=True)
    proc.wait()

    # --- upload results CSV to S3 ---
    results_s3 = config.get("results_s3_output")
    if results_s3:
        output_dir = f"/tmp/navsim_exp/{experiment_name}"
        for csv_name in (f"navtest_v{dataset_version}.csv", f"navtest_v{dataset_version}_bug_fix.csv"):
            csv_path = os.path.join(output_dir, csv_name)
            if os.path.exists(csv_path):
                _upload_file(csv_path, results_s3, csv_name)
                logger.info("Results uploaded to %s/%s", results_s3, csv_name)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # --- optional: save computed caches to S3 ---
    if config.get("test_cache_s3_output") and not test_cache_s3:
        _upload_cache(TEST_CACHE_LOCAL, config["test_cache_s3_output"])
    if config.get("metric_cache_s3_output") and not metric_cache_s3:
        _upload_cache(METRIC_CACHE_LOCAL, config["metric_cache_s3_output"])


def _compute_test_cache(config, repo_root, dataset_version):
    """Compute test feature cache by running the training script on navtest with cache computation."""
    dataset_s3 = config["dataset_s3_path"]
    logger.info("Computing test feature cache from raw data at %s ...", dataset_s3)
    # Download navtest sensor data (needed to compute features)
    for sub in ("navsim_logs/test", "sensor_blobs/test"):
        download_s3_prefix(
            f"{dataset_s3.rstrip('/')}/{sub}",
            os.path.join(DATASET_LOCAL, sub),
            endpoint_url=RESEARCH_DATASETS_ENDPOINT,
            region=RESEARCH_DATASETS_REGION,
        )
    # Run training script in cache-compute mode on navtest split
    script = os.path.join(repo_root, TRAINING_SCRIPT)
    cmd = [
        sys.executable, script,
        "--config-name", "default_training",
        "agent=sparsedrive_agent",
        "experiment_name=test_cache_computation",
        "train_test_split=navtest",
        "use_cache_without_dataset=False",
        "force_cache_computation=True",
        f"cache_path={TEST_CACHE_LOCAL}",
        "trainer.params.max_epochs=0",   # no training, just cache computation
        "trainer.params.fast_dev_run=false",
        "trainer.params.limit_train_batches=0",
        "trainer.params.devices=1",
    ]
    _run_cmd(cmd, tag="cache")


def _compute_metric_cache(config, repo_root, dataset_version):
    """Compute PDM metric cache using run_metric_caching.py."""
    dataset_s3 = config["dataset_s3_path"]
    logger.info("Computing metric cache from raw data at %s ...", dataset_s3)
    for sub in ("navsim_logs/test", "sensor_blobs/test", "maps"):
        download_s3_prefix(
            f"{dataset_s3.rstrip('/')}/{sub}",
            os.path.join(DATASET_LOCAL, sub),
            endpoint_url=RESEARCH_DATASETS_ENDPOINT,
            region=RESEARCH_DATASETS_REGION,
        )
    script = os.path.join(repo_root, METRIC_CACHING_SCRIPTS[dataset_version])
    cmd = [
        sys.executable, script,
        "--config-name", "default_metric_caching" if dataset_version == "v2" else "default_metric_caching_v1",
        "train_test_split=navtest",
        f"metric_cache_path={METRIC_CACHE_LOCAL}",
    ]
    _run_cmd(cmd, tag="metric_cache")


def _run_cmd(cmd, tag="cmd"):
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=os.environ,
    )
    for line in proc.stdout:
        print(f"[{tag}] {line.rstrip()}", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _upload_file(local_path, s3_prefix, filename):
    """Upload a single file to S3."""
    import boto3, botocore.config
    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-chicago-1")
    cfg = botocore.config.Config(
        connect_timeout=30, read_timeout=300, signature_version="s3v4",
        request_checksum_calculation="when_required", response_checksum_validation="when_required",
    )
    s3 = boto3.client("s3", endpoint_url=endpoint, region_name=region, config=cfg)
    path = s3_prefix[5:]
    bucket, key_prefix = path.split("/", 1)
    s3.upload_file(local_path, bucket, f"{key_prefix.rstrip('/')}/{filename}")
