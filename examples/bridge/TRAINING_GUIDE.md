# Bridge data preprocessing and Pi0 finetuning

This guide covers the complete pipeline from converting LeRobot dataset to training PI0 model, with Allenren's open-pi-zero preprocessing (quantile normalization, camera setup, delta actions) properly applied.

## Step 1: Download the Bridge TFDS data and convert to LeRobot format

- Download source: `https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/`
- (Optional) Preprocess to resize to 224 x 224 for Paligemma, following the instructions,
  `https://github.com/allenzren/open-pi-zero?tab=readme-ov-file#data-pre-processing`

Convert Bridge TFDS to LeRobot format using (assuming raw data at \path_to_bridge\bridge_dataset):
```bash
uv run --group rlds examples/bridge/convert_bridge_to_lerobot.py \
    --data_dir /path_to_bridge \
    --output_dir /path_to_output \
    --repo_name bridge_lerobot_224
```

### Verify Converted Dataset

Check that your converted dataset exists and has the correct structure:

```bash
# Default location (HuggingFace cache)           
ls ~/.cache/huggingface/lerobot/bridge_lerobot_224/

# Or custom location if you used --output_dir in Step 0
ls /path_to_output/bridge_lerobot_224/
```

You should see:
```
bridge_lerobot_224/
├── data/           # Parquet files with episode data (includes images)
├── meta/           # Metadata (info.json, tasks.jsonl, stats/)
```

Verify the dataset loads correctly:
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("bridge_lerobot_224")
print(f"Total episodes: {dataset.num_episodes}")
print(f"Total frames: {dataset.num_frames}")
print(f"Features: {list(dataset.features.keys())}")
# Expected: ['image_0', 'image_1', 'state', 'actions', ...]
```

## Step 2: Compute Normalization Statistics

**CRITICAL**: The following things are configured in `src/openpi/training/config.py`
- Config name: `pi0_bridge_finetune`
- Dataset repo_id: `bridge_lerobot_224`
- Bridge uses **quantile normalization** (1st-99th percentile → [-1, 1]), not z-score normalizatin.

```bash
export HF_DATASETS_CACHE=/data/jerry/datasets/openx/cache/datasets
export HF_LEROBOT_HOME=/data/jerry/datasets/openx

# max-frames is optional, but default value (using full dataset) takes a long time
uv run python scripts/compute_norm_stats.py --config-name pi0_bridge_finetune --max-frames 100000
```

**What this does:**
- With max-frames specified, only iterates through this number of frames of the converted Bridge dataset
- Computes normalization statistics for `state` and `actions`
- **Mean/std**: For reference (not used during training with quantile norm)
- **q01/q99**: 1st and 99th percentile values (used for quantile normalization)
- Saves to: `assets/pi0_bridge_finetune/bridge_lerobot_224/norm_stats.json`
- Cache: for the first time running `compute_norm_stats.py`, it also generates Arrow files of the same size as the lerobot `data` (from Step 1) and caches in `HF_DATASETS_CACHE`.

You can verify the stats were computed correctly:
```bash
# Check file exists
ls -lh assets/pi0_bridge_finetune/bridge_lerobot_224/norm_stats.json

# View contents
cat assets/pi0_bridge_finetune/bridge_lerobot_224/norm_stats.json
```

## Step 3: Understand the Training Config

The `pi0_bridge_finetune` config in `src/openpi/training/config.py` has incorporated Allenren/open-pi-zero's preprocessing:

**Key config settings:**
```python
LeRobotBridgeDataConfig(
    repo_id="bridge_lerobot_224",  # dataset name (must match the --repo_name used in Step 0)
    camera_keys=("image_0", "image_1"),  # Primary + secondary (NOT wrist)
    assets=AssetsConfig(asset_id="bridge_lerobot_224"),  # MUST match repo_id for norm stats path
)
# Note: use_quantile_norm=True is automatically set for Bridge in the config
```

**Preprocessing pipeline:**
1. **Camera mapping**: `image_0` → primary, `image_1` → secondary
2. **Delta actions**: Absolute EE positions converted to delta actions
3. **Quantile normalization**: Maps q01-q99 to [-1, 1] for both state and actions
4. **Action padding**: 7D actions padded to 32D (PI0 requirement)

## Step 4: Launch Default Training

The default training of Pi0 requires both the lerobot dataset (from Step 1) and the cached Arrow files (from Step 2), as well as the `norm_stats.json` from Step 2. 

Set environment variables (same as Step 2):
```bash
export HF_DATASETS_CACHE=/data/jerry/datasets/openx/cache/datasets
export HF_LEROBOT_HOME=/data/jerry/datasets/openx
```

### Option A: Single GPU Training

```bash
uv run python scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name my_bridge_experiment
```

### Option B: Multi-GPU Training (Recommended for 60K episodes)

```bash
# Use all available GPUs (replace 2 with your GPU count)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name my_bridge_experiment
```

### Option C: Resume from Checkpoint

```bash
uv run python scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name my_bridge_experiment \
    --resume
```

### Verify training scripts
`./verify_training.sh`

## Step 4 (Alternative): Lauch Cache-only Training

Since the cached Arrow files under `HF_DATASETS_CACHE` is what will actually be used, `data` under `bridge_lerobot_224` can be bypassed by some overriding of `LeRobotDataset`.
For cache-only training, only `bridge_lerobot_224/meta` and cached Arrow files under `HF_DATASETS_CACHE` are needed.

### Verify training scripts
`./verify_training_cache_only.sh`

## Step 5: Launch Training on Cluster 

**Required files on cluster (refer to job_scripts for env vars' values)**
1. **bridge_lerobot data and meta directories**: under `HF_LEROBOT_HOME`
1. **cached Arrow data**: under `HF_DATASETS_CACHE`
2. **pi0_base_pytorch_weights**: under `PYTORCH_WEIGHT_PATH`
3. **big_vision tokenizer**: under `OPENPI_DATA_HOME`
4. **normalization stats**: `assets` dir under the openpi root, will be synced automatically

**hobot_submission config**
Set `code_dirs` to include the openpi path in `hobot_submission/conf.py`

### Default cluster training

```bash
python -m cluster_train --job_script openpi/examples/bridge/job_openpi_bridge.sh --root_dir /local_tb_sync_dir -s 17 -g 0,1,2,3,4,5,6,7
```

### Cache-only cluster training
```bash
python -m cluster_train --job_script openpi/examples/bridge/job_openpi_bridge_cache_only.sh --root_dir /local_tb_sync_dir -s 18 -g 0,1,2,3,4,5,6,7
```

**Estimated training time:**
- 8× H20 GPUs: ~3.5 days for 100K train steps

## Customize Training

### Training Configuration Details

**From `pi0_bridge_finetune` config:**

```python
# Model
model = pi0_config.Pi0Config(
    action_dim=32,       # Padded from 7D Bridge actions
    action_horizon=16,   # Predict 16 future actions
)

# Learning rate schedule
lr_schedule = CosineDecaySchedule(
    warmup_steps=1_000,
    peak_lr=3e-5,       # Peak learning rate
    decay_steps=50_000,
    decay_lr=3e-6,      # Final learning rate
)

# Training parameters
num_train_steps=50_000
batch_size=128
num_workers=2  # Parallel data loading workers
```

You can also override config parameters via command line:

### Change batch size
```bash
uv run python scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name my_experiment \
    --batch-size 64  # Reduce if OOM
```

### Change learning rate
```bash
# Note: Learning rate cannot be overridden from CLI
# Edit the config in src/openpi/training/config.py if needed
```

### Change number of training steps
```bash
uv run python scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name my_experiment \
    --num-train-steps 100000
```

### Use different dataset name
```bash
# Note: Dataset repo_id cannot be overridden from CLI
# Edit the config in src/openpi/training/config.py if needed
```

### Change checkpoint save interval
```bash
uv run python scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name my_experiment \
    --save-interval 2500  # Save every 2500 steps instead of 5000
```

## Troubleshooting

### Error: "Normalization stats not found"
**Cause**: Haven't run `compute_norm_stats.py` yet
**Fix**: Run Step 2 above

### Error: "api_key not configured (no-tty). call wandb.login"
**Cause**: You enabled WandB with `--wandb-enabled` but haven't logged in
**Fix**: Configure WandB login:
```bash
wandb login [your_api_key]
```

**Note**: WandB is disabled by default, so you shouldn't see this error unless you explicitly enable it.

### Error: "Norm stats not found in assets/pi0_bridge_finetune/bridge"
**Cause**: The `asset_id` in config doesn't match your dataset `repo_id`
**Fix**: The `asset_id` should match your `repo_id`. Edit `src/openpi/training/config.py` line 1077:
```python
assets=AssetsConfig(asset_id="bridge_lerobot_224"),  # Match your repo_id
```

### Error: "Dataset 'bridge_lerobot_224' not found"
**Cause**: Dataset repo_id doesn't match converted dataset name
**Fix**: Either:
- Reconvert with matching name: `--repo_name bridge_lerobot_224`
- Or override in training: `--config.data.repo_id your_actual_name`
- Or check your environment variables are set correctly (HF_LEROBOT_HOME)

### Error: "quantile stats must be provided"
**Cause**: Normalization stats don't have q01/q99 values
**Fix**: Re-run `compute_norm_stats.py` to regenerate stats

### OOM (Out of Memory) during training
**Fix**: Reduce batch size
```bash
uv run python scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name my_experiment \
    --batch-size 64  # or 32
```

### Slow data loading
**Fix**: Increase number of workers
```bash
uv run python scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name my_experiment \
    --num-workers 4
```
