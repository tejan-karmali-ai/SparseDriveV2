# Installation for SparseDriveV2

After successfully installing the NAVSIM environment, you should further proceed to install the following packages for SparseDriveV2:

```bash
conda activate navsim
cd navsim/agents/sparsedrive/ops
python setup.py develop
```

# Data and metric caching

```bash
sh scripts/cache/run_dataset_caching_navtrain.sh
sh scripts/cache/run_dataset_caching_navtest.sh

# for navsimv1
sh scripts/cache/run_metric_caching_navtrain_v1.sh
sh scripts/cache/run_metric_caching_navtest_v1.sh

# for navsimv2
sh scripts/cache/run_metric_caching_navtrain_v2.sh
sh scripts/cache/run_metric_caching_navtest_v2.sh
```

# Anchor preparation
You can download path/velocity/trajectory anchor files from [here](https://huggingface.co/wenchaosun/SparseDriveV2) and put to ckpt/kmeans/ or cluster by
```bash
mkdir -p ckpt/kmeans
sh scripts/cluster/cluster_anchor.py
```

# Checkpoint
Download [resnet-34 backbone](https://huggingface.co/timm/resnet34.a1_in1k/blob/main/pytorch_model.bin) and put to ckpt/resnet34.bin. Download pretrained weight from [here](https://huggingface.co/wenchaosun/SparseDriveV2).


# Training
```bash
# navsimv1
sh scripts/training/sparsedrive_navsimv1.sh
# navsimv2
sh scripts/training/sparsedrive_navsimv2.sh
```

# Evaluation
```bash
# navsimv1
sh scripts/evaluation/run_pdm_score_navtest_v1.sh
# navsimv2
sh scripts/evaluation/run_pdm_score_navtest_v2.sh
```

The EPDM scores for navsimv2 both before and after the [bug fix](https://github.com/autonomousvision/navsim/issues/151#issue-3379282167) will be reported.

## Evaluation protocol versions

The evaluation script produces results under three named protocols. All three run on the same scene set (396 navmini or navtest scenarios); they differ only in scoring logic.

### navmini_v1 / navtest_v1
Uses the **NAVSIMv1** framework (`navsim.navsim_v1.*`). Output: `navtest_v1.csv`.

- Metrics: `no_at_fault_collisions`, `drivable_area_compliance`, `driving_direction_compliance`, `time_to_collision_within_bound`, `comfort`, `ego_progress`
- Non-reactive traffic agents; no two-frame extended comfort
- Final score: **PDMS**

### navmini_v2 / navtest_v2
Uses the **NAVSIMv2** framework. Output: `navtest_v2.csv`.

- Adds `lane_keeping`, `history_comfort`, `two_frame_extended_comfort`, `traffic_light_compliance`
- Reactive traffic agents during simulation
- Human-penalty filter: if the human trajectory also fails a metric, the model is not penalized (that metric is set to 1)
- Final score: **EPDMS**
- **Known bug**: after the human-penalty filter modifies individual metric values, `multiplicative_metrics_prod` (product of binary safety metrics) and the `weighted_metrics` array are not recomputed. This means those intermediate values go stale and the final EPDMS is computed from inconsistent state.

### navmini_v2_bugfix / navtest_v2_bug_fix
Same as v2 but uses `navsim/evaluate/pdm_score_fix_bug.py`. Output: `navtest_v2_bug_fix.csv`.

- Adds a recalculation step after the human-penalty filter: recomputes `multiplicative_metrics_prod` and `weighted_metrics` so they stay consistent with the corrected per-metric values before the final score is aggregated.
- Scores differ from v2 only for scenarios where the human trajectory also fails at least one metric; in those cases bugfix scores are equal or higher.
- This is the **recommended** protocol for reporting NAVSIMv2 results.

