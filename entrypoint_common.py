"""Common utilities for Lilypad entrypoints."""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_environment(install_requirements=True, install_local_pkg=True):
    """Install requirements and the local navsim package inside the Lilypad container."""
    if install_requirements:
        logger.info("Installing requirements from requirements_lilypad.txt...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements_lilypad.txt"],
            check=True,
        )
    if install_local_pkg:
        logger.info("Installing local navsim package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
            check=True,
        )
    _prepend_lib_paths()


def download_s3_prefix(s3_path, local_dir, endpoint_url=None, region=None):
    """Sync an S3 prefix to a local directory.

    Args:
        s3_path: s3://bucket/prefix URI.
        local_dir: Destination directory on the local filesystem.
        endpoint_url: Override the S3-compatible endpoint (e.g. for OCI Phoenix).
            Defaults to AWS_ENDPOINT_URL_S3 env var.
        region: Override AWS_DEFAULT_REGION.
    """
    import boto3
    import botocore.config

    os.makedirs(local_dir, exist_ok=True)
    bucket, key_prefix = _parse_s3_uri(s3_path)
    s3 = _get_s3_client(endpoint_url=endpoint_url, region=region)

    paginator = s3.get_paginator("list_objects_v2")
    prefix = key_prefix.rstrip("/") + "/"
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel:
                continue
            local_path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not os.path.exists(local_path):
                s3.download_file(bucket, obj["Key"], local_path)
                count += 1
    logger.info("Downloaded %d files from %s to %s", count, s3_path, local_dir)
    return local_dir


def download_s3_archive(s3_path, local_dir):
    """Download a .tar or .tar.gz archive from S3 and extract it."""
    import tarfile

    os.makedirs(local_dir, exist_ok=True)
    bucket, key = _parse_s3_uri(s3_path)
    s3 = _get_s3_client()
    tar_local = f"/tmp/{s3_path.split('/')[-1]}"
    if not os.path.exists(tar_local):
        logger.info("Downloading archive %s...", s3_path)
        s3.download_file(bucket, key, tar_local)
    with tarfile.open(tar_local, "r:*") as tar:
        tar.extractall(local_dir)
    os.remove(tar_local)
    return local_dir


def download_hf_checkpoint(url, local_path):
    """Download a checkpoint from a HuggingFace URL via wget."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        logger.info("Checkpoint already exists at %s, skipping download.", local_path)
        return local_path
    logger.info("Downloading checkpoint from %s...", url)
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", local_path, url],
        check=True,
    )
    return local_path


def _get_s3_client(endpoint_url=None, region=None):
    import boto3
    import botocore.config

    endpoint = endpoint_url or os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-chicago-1")
    config = botocore.config.Config(
        connect_timeout=30,
        read_timeout=300,
        signature_version="s3v4",
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )
    # Use Phoenix-specific credentials when connecting to the OCI Phoenix endpoint.
    # Set OCI_PHX_ACCESS_KEY_ID / OCI_PHX_SECRET_ACCESS_KEY in the environment when
    # the Phoenix and Chicago OCI tenancies use different Customer Secret Keys.
    if endpoint and "phoenix" in endpoint and os.environ.get("OCI_PHX_ACCESS_KEY_ID"):
        access_key = os.environ["OCI_PHX_ACCESS_KEY_ID"]
        secret_key = os.environ["OCI_PHX_SECRET_ACCESS_KEY"]
    else:
        access_key = None
        secret_key = None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=config,
    )


def _parse_s3_uri(s3_path):
    path = s3_path[5:]  # strip "s3://"
    bucket, *key_parts = path.split("/")
    return bucket, "/".join(key_parts)


def _prepend_lib_paths():
    """Add torch/lib and NVIDIA vendor libs to LD_LIBRARY_PATH."""
    try:
        import site
        import torch

        paths = [str(Path(torch.__file__).parent / "lib")]
        for sp in site.getsitepackages():
            for sub in ("nvidia/nccl/lib", "nvidia/cublas/lib", "nvidia/cudnn/lib"):
                candidate = Path(sp) / sub
                if candidate.exists():
                    paths.append(str(candidate))
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        prefix = ":".join(p for p in paths if Path(p).exists())
        if prefix:
            os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix
    except Exception:
        pass
