# GraspCVAE on DexGYSNet

Reproduction of the GraspCVAE baseline of Table 1 in *Grasp as You Say* (DexGYS, NeurIPS 2024, arXiv 2405.19291)
on the DexGYSNet dataset with the Shadow hand. GraspTTA, which builds on this model, is in the `grasptta` repo.

## Results

Scored like our DexGYSGrasp reproduction (dexter_temp DexGYS benchmark; diversity = 8 seeded samplings per
input condition). Con. and Q1 are raw × 10 (the paper's ×10⁻¹ unit).

| Method | P-FID ↓ | CD ↓ | Con. ↓ | SR sim-only ↑ | SR E_pen² < 0.001 ↑ | SR max_pen < 5mm ↑ | Q1 ↑ | Pen. (cm) ↓ | δt (cm) | δr (deg) | δq (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GraspCVAE (paper) | 29.02 | 3.14 | 0.96 | 29.12 * | | | 0.54 | 0.55 | 0.18 | 1.76 | 0.18 |
| **GraspCVAE (ours)** | 21.00 | 2.80 | 0.62 | 48.08 | 44.73 | 27.16 | 0.93 | 0.38 | 0.22 | 3.47 | 0.53 |

\* the paper reports one Success rate without saying which penetration filter it uses.

## Checkpoint

Private Hugging Face repo [`EunhaPark/graspcvae`](https://huggingface.co/EunhaPark/graspcvae) (log in with a token
that has access). Paths in the repo mirror `Experiments/`, so the commands below work unchanged after:

```bash
hf auth login   # once
hf download EunhaPark/graspcvae --local-dir ./Experiments
```

| Checkpoint (`EunhaPark/graspcvae`) | Config | Notes |
|---|---|---|
| `graspcvae_kl10/epoch200_minus_loss_-5.0587_latest.pth` (1.7 GB) | `config/graspcvae.yaml` + `model.criterion.loss_weights.kl 10.0` | β = 10, hand-point encoder, epoch 200 |

The checkpoint is 1.7 GB because it includes the frozen CLIP ViT-L/14 weights.

## Environment

The Grasp-as-You-Say image (`Dockerfile.gys` in Grasp-as-You-Say; `postech-a.kr-central-2.kcr.dev/dmsgk724/grasp-as-you-say:latest`)
for training / test, and the dexter image (`.../dmsgk724/dexter:87c1e1d`) for the benchmark metrics.
`data/` must point to Grasp-as-You-Say's `data/` (DexGYSNet json, OakInk shapes, Shadow hand MJCF); the committed
symlink expects Grasp-as-You-Say two levels up (`../../Grasp-as-You-Say/data`). Run everything from the repo root.

## Evaluation commands

### 1. Generate grasps on the test set

```bash
# z ~ N(0, I), seed 3407
python test.py --train_cfg ./config/graspcvae.yaml --test_cfg ./config/infer_graspcvae_test.yaml \
  --override model.checkpoint_path "'./Experiments/graspcvae_kl10/epoch200_minus_loss_-5.0587_latest.pth'" \
             save_root "'./Experiments/my_eval/graspcvae'" seed 3407
```
Writes `<save_root>/test_results/epoch_200_test/{raw_results,matched_results}.json` (~2 min on one GPU).
`save_root` must not exist yet, otherwise the script asks before deleting it.

### 2. Convert to the dexter benchmark format

```bash
python scripts/to_dexter_predictions.py \
  --matched ./Experiments/my_eval/graspcvae/test_results/epoch_200_test/matched_results.json \
  --out-dir ./Experiments/my_eval/graspcvae_pred
# -> graspcvae_pred/dexter309/predictions.json   (same rule as our DexGYSGrasp reproduction)
```

### 3. Metrics: P-FID, CD, Con., Q1, Pen. (dexter image)

```bash
source /python/.venv/bin/activate
cd /workspace/users/dmsgk724/dexter_temp && uv pip install -e .
PRED=<this repo>/Experiments/my_eval/graspcvae_pred/dexter309/predictions.json
DATA=/workspace/datasets/dexterp/dexgys_final
python -m benchmark.dexgys.chamfer --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
python -m benchmark.dexgys.q1      --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
python -m benchmark.dexgys.fid     --pred-path $PRED --data-path $DATA
# -> metrics.json next to predictions.json (cmap and valid_q1 raw; mean_pen in m)
```
`kubeflow/run_eval_metrics.sh` does steps 2–3 for a whole run (`SAVE_ROOT`, `NAME`, `VARIANTS` env vars).

### 4. Success rate (Isaac Gym, dexter_temp `.venv-isaacgym`)

```bash
bash benchmark/dexgys/setup_isaacgym.sh && source .venv-isaacgym/bin/activate   # once, in dexter_temp
python benchmark/dexgys/success_rate.py --pred-path $PRED --data-path $DATA --no_force --gpu 0 --parallel 8
```
Prediction file of the reported run (cholab PVC):
`/workspace/users/dmsgk724/dexgys_baseline_work/graspcvae/Experiments/eval/graspcvae_kl10_latest/dexter309/predictions.json`

### 5. Diversity (δt, δr, δq)

```bash
# 8 samplings (seed 3407..3414) of the latest checkpoint of the run in SAVE_ROOT
SAVE_ROOT=./Experiments/graspcvae_kl10 NAME=graspcvae_kl10 VARIANTS=latest bash kubeflow/run_diversity.sh
# -> Experiments/eval/graspcvae_kl10_latest/diversity/dexter309/diversity_with_std.json  (mean_object = reported)
```

## Training

```bash
# beta = 10 (the config default kl is 0.01, so override it)
python train.py -t ./config/graspcvae.yaml --override save_root "'./Experiments/graspcvae_kl10'" model.criterion.loss_weights.kl 10.0
```
200 epochs take ~13 h on one GPU.

Kubeflow manifests of the reported run (job names end with `-eunha`; they run from the cholab PVC working copy
`dexgys_baseline_work/graspcvae`): `kubeflow/graspcvae/trainjob-graspcvae.yaml` (train + test),
`kubeflow/graspcvae/evaljob-graspcvae-{metrics,diversity}.yaml`.

## How it was built

CVAE from GraspTTA's released code (`network/CVAE.py`, `affordanceNet`: encoder MLP → μ, log σ² of a 64-d latent;
decoder MLP), with the DexGYS encoders the paper asks for ("same encoder structure"): object PointNet++
(IDGC config) and CLIP ViT-L/14 + adapter. Decoder input = [z ; f_obj ; f_lan] (paper A.3.3). Loss = DexGYS Eq.5
(`GraspLossPose`, weights para 10, chamfer 1, penetration 100, self-penetration 10, cmap 10) + β·KL.
Not specified by the paper, chosen here:
- **KL weight β = 10**: the posterior collapses (training KL → 0), which gives the near-zero diversity the paper
  reports for GraspCVAE.
- **CVAE encoder input**: GT hand surface points (1024, Shadow FK) → PointNet++ (QGC `hand_backbone`) → max-pool,
  as GraspTTA encodes hand vertices.
- **Pooling**: PointNet++ local features max-pooled; language = CLIP EOS token; condition c = 1024-d.
- **Output**: normalized Shadow pose, 28-d (translation 3, joints 22, Euler 3).
- **Optimization**: identical to our DexGYSGrasp IDGC run (200 epochs, batch 48, Adam 1e-4 → 1e-5 cosine, wd 1e-4);
  the paper's Sec 5.2 values (batch 64, lr 2e-4) differ from its released config.
- **Checkpoint**: latest (epoch 200).
