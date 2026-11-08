# Reproducing the experiments

Full instructions for the twelve experiment families in the paper: data and
reference preparation, every entry point and configuration, and the expected
outputs. Run the commands from the repository root. For installation and a quick
start, see the [README](../README.md). Table and figure numbers below follow the
version of the paper at the time of release and may differ in later revisions.
No trained weights, model banks, or generated figures are bundled.

## Installation

Use Python 3.10 or 3.11 and PyTorch 2.5.1 or newer. Install a CUDA-specific PyTorch wheel
appropriate for your device if needed, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps .
python -m dsprobe --help
```

No MVProbe or ProbeX checkout is required: the necessary model, data-loader,
training, and aggregation code is included. Optional dependencies:

```bash
python -m pip install '.[download]'  # public-bank / pretrained-reference acquisition
python -m pip install '.[text]'      # CLIP text-feature preparation for SD-LoRA
python -m pip install '.[statnn]'    # XGBoost 3.2.0 for the CPU StatNN baseline
```

The base requirements do not install XGBoost; only StatNN needs the `statnn`
extra. It also installs scikit-learn 1.7.2, required by `XGBClassifier` but not
installed by bare XGBoost. StatNN requires neither `transformers` nor an
upstream source checkout.

The text extra pins the public OpenAI CLIP repository. Alternatively, install
an explicit local checkout at the relative path `external/CLIP`:

```bash
git clone https://github.com/openai/CLIP.git external/CLIP
git -C external/CLIP checkout d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
python -m pip install ./external/CLIP
```

The baseline sources are [MVProbe](https://github.com/AI-hew-math/MVProbe) and
[ProbeX](https://github.com/eliahuhorwitz/ProbeX). Optional upstream checkouts
may be placed at `external/MVProbe` and `external/ProbeX`; the runners do not
import from them. Required third-party attribution is retained in
`THIRD_PARTY_NOTICES.md` and `LICENSE`.

## Experiment map

| # | Experiment | Paper location | Entry point / configuration |
|---|---|---|---|
| 1 | Single-layer fixed | Table 2 | `train --suite configs/modelj_fixed.json` |
| 2 | SD-LoRA ID / zero-shot | Table 2 | `train --suite configs/sd_lora.json` |
| 3 | Cross-tree shared encoder | Table 4 | `python -m dsprobe.shared` |
| 4 | MVProbe higher-order / capacity controls | Table 6 | `train --suite configs/higher_order.json` |
| 5 | Independent single-response branches | Table 10 | `train --suite configs/single_branch.json` |
| 6 | Layer-wise sweep | Figures 1 and 3, Table 12 | `train --suite configs/modelj_layerwise.json` |
| 7 | Free-offset control | Table 11 | `python -m dsprobe.offset` |
| 8 | Reference choice | Table 3 | `train --suite configs/reference_choice.json` |
| 9 | Reference initialization | Table 13 | `train --suite configs/reference_init.json --learned-init ...` |
| 10 | Probe-count sweep | Figure 2, Table 9 | `python -m dsprobe.probe_sweep` |
| 11 | Parameter and FLOP accounting | Tables 7 and 8 | `python -m dsprobe costs --output results/costs` |
| 12 | StatNN baseline | Table 2, StatNN row | `train --suite configs/statnn.json` |

Tables 3, 11, 13-15 are additional ablations in the **single-layer fixed setting**:
each run probes one specified layer rather than sweeping the network. Every one
of them has an entry point:

| Table | What varies | Command |
|---|---|---|
| 3 | the reference that is subtracted | `train --suite configs/reference_choice.json` |
| 11 | offset parameterization | `python -m dsprobe.offset` |
| 13 | where the learned reference starts | `train --suite configs/reference_init.json --learned-init {mean,pretrained,random,random_norm}` |
| 15 | the reference-update cap | `train --suite configs/modelj_fixed.json --methods r_dsprobe --cap VALUE` |

Table 14 reports the distance between the learned reference and the pretrained
weight. Train with `--save-weights`, then aggregate:

```bash
python -m dsprobe train --suite configs/modelj_fixed.json --methods r_dsprobe \
  --data-root data --reference-root references --output results/saved \
  --save-weights --device cuda:0
python -m dsprobe distance --results results/saved \
  --data-root data --reference-root references --output results/distance
```

`distance` reads the `x_star` of each seed's validation-selected state and
reports `||W' - W_pre||_F / ||W_pre||_F * 100` for both the initialization
(the training-bank mean) and the learned reference, with the change between
them. `--save-weights` is required: resuming a run that did not save states
cannot recover them.

Default seeds are 1–5. Training and aggregation use actual runs, not embedded
paper values.

### Probe-count sweep

`dsprobe.probe_sweep` runs `configs/probe_sweep.json` once per probe count and
writes each count to its own subdirectory, so the results do not collide:

```bash
python -m dsprobe.probe_sweep --probes 1,2,4,8,16,32,64,128 \
    --data-root data --reference-root references --output results/probe_sweep \
    --device cuda:0
python -m dsprobe.probe_sweep --output results/probe_sweep --summarize
```

`--summarize` writes `probe_sweep_summary.csv` with the mean and population
standard deviation over seeds for each (tree, method, r), the trainable-parameter
count, and `forward_flops_per_example`. This is the forward matrix-multiplication
count including the weight responses, projections, encoder, prediction head,
and reference responses. R-DSProbe adds `12*m*n*r/B` FLOPs per example because
its six reference products are computed once per batch. One multiply-add counts
as two FLOPs; normalization, elementwise operations, losses, and backward are
outside this forward matrix-multiplication metric.

The summary reads the matrix shape, probe count, projection/representation
dimensions, class count, and nominal batch size from each saved run's contract.
`--flops-batch-size 128` can override the accounting batch size when summarizing
existing results; it does not change training. A compatible completed MVProbe
run at r=128 supplies `mvprobe_128_flops_per_example` and
`reduction_vs_mvprobe_128`. These columns are blank when no compatible baseline
is present. Duplicate seeds or conflicting cost configurations are rejected.
Rerun `--summarize` on existing result files to regenerate the corrected CSV;
no retraining is needed for this accounting change.

`--probes`, `--models`, `--methods` and `--seeds` all accept comma-separated
lists, and `--dry-run` prints the job list without training. The default probe
grid is unchanged. To measure the paper's additional comparison points, include
6, 24, and 96 explicitly with `--probes 1,2,4,6,8,16,24,32,64,96,128`.

### Parameter and FLOP accounting

Tables 7 and 8 are produced without training or data:

```bash
python -m dsprobe costs --output results/costs --batch-size 128
```

The command also writes `probe_efficiency.csv` and `probe_efficiency.json` for
MVProbe at r=128 and R-DSProbe at r=6/24/32/96 for ResNet/SupViT/MAE/DINO.
Each row reports `mvprobe_flops_per_example`, `r_dsprobe_flops_per_example`
(including reference responses), and their ratio `reduction`. Counts are raw
FLOPs per example; divide by 1e6 to display M FLOPs. The batch size is recorded,
and changing `--batch-size` changes the amortized reference cost. These are
symbolic costs for the paper-specified probe counts, not newly measured accuracy
or automatically selected matching probe counts. No model bank is required.

It writes `costs.csv` and `costs.json` for MVProbe and the Gram-difference model
at the accounting layers (ResNet 55, SupViT 59, MAE 65, DINO 47): allocated and
trainable parameters, the size of the learned reference, and symbolic
forward/backward multiply-add FLOPs, with one multiply-add counted as two FLOPs.
`costs.json` records what the count excludes.

Cost depends on the probed matrix only through its shape, so the accounting
layer is chosen for shape rather than accuracy. SupViT and MAE are $768\times768$
at either method's selected layer (53 or 59, 65 or 64), so the numbers are the
same either way. ResNet is the exception: L98 is $2048\times512$ against L55's
$1024\times256$, and the larger matrix inflates the reference $W_*$ --- the very
quantity the table measures --- without either method gaining anything. ResNet is
therefore accounted at L55, the same shape-matched choice the probe-count
comparison makes. `costs.json` states this in its `layer_selection` field.

The paper's training-FLOP column is `estimated_3x_forward_flops_per_example`
(one forward, two backward). The CSV also carries a counted autograd backward in
`actual_train_flops_per_example`, which is lower; both are emitted so the
estimate can be checked against the count. The backward figure is a counted
autograd pass, not the usual "three times forward" estimate; both are reported
so they can be compared.

### StatNN baseline

```bash
python -m pip install '.[statnn]'
python -m dsprobe train --suite configs/statnn.json \
  --data-root data --output results/statnn
python -m dsprobe analyze --results results/statnn --output results/statnn_summary
```

StatNN uses **all named parameter tensors of each checkpoint**, not a selected
layer. For each tensor, flattened independently, it computes seven features in
this order: **q0, q25, q50, q75, q100, population standard deviation (`ddof=0`),
mean**. Features are concatenated in sorted parameter-name order. This includes
weights and biases, but excludes buffers such as BatchNorm running statistics;
checkpoint metadata supplies target labels, never input features. There is no
MLP, learned feature normalizer, reference subtraction, or layer-selection step.

| Architecture | Named parameter tensors | Features per model |
|---|---:|---:|
| ResNet | 314 | 2,198 |
| SupViT / MAE / DINO | 200 each | 1,400 each |

`dsprobe/statnn.py` reads and validates parameter names and shapes against the
packaged audited manifest `dsprobe/assets/statnn_parameters.json`. It does not
instantiate a target architecture or import `transformers` or upstream code.
The seven-statistic definition follows Unterthiner et al. (2020),
[Google Research's `dnn_predict_accuracy` notebook](https://github.com/google-research/google-research/blob/60a55a0fb30eb0722fa49aa7bb3412312d57c314/dnn_predict_accuracy/colab/dnn_predict_accuracy.ipynb)
at commit `60a55a0fb30eb0722fa49aa7bb3412312d57c314`. This is an independently
implemented **Model-J adaptation with a 100-output XGBoost classifier**, not the
original LightGBM accuracy-regression experiment.

The fixed classifier settings are defined in `configs/statnn.json`: XGBoost
3.2.0, at most 500 boosting rounds, early stopping after 25 rounds without
validation-logloss improvement, learning rate 0.05, maximum depth 3,
`subsample=0.9`, `colsample_bytree=1.0`, `reg_alpha=0`, `reg_lambda=1`,
`tree_method="hist"`, `device="cpu"`, and `n_jobs=4`. The objective is
`binary:logistic`, with `eval_metric="logloss"` and
`multi_strategy="one_output_per_tree"`. Seeds are 1–5. The validation set selects
the boosting iteration; the test set is evaluated after selection, never used
for early stopping or hyperparameter selection. Reported accuracy is the
percentage of correct binary class-membership decisions across all test models
and 100 classes, using a probability threshold of 0.5. Aggregation reports the
five-seed mean and population standard deviation (`ddof=0`), with the layer
field set to `full_model`. No previously reported numerical result is embedded
or claimed as newly reproduced by these commands.

StatNN defaults to CPU; `--device cpu` is optional, and an explicit CUDA device
is rejected. `--models`, `--seeds`, `--resume`, and `--dry-run` are supported.
`--num-workers` controls feature-extraction workers (default 2), while
`--threads` controls PyTorch threads (default 2); neither overrides XGBoost's
fixed `n_jobs=4`. Neural-network-only flags, including `--layers`,
`--model-layers`, `--epochs`, `--learning-rate`/`--lr`, `--n-probes`/`--k`, and
projection/representation dimensions, are rejected instead of silently ignored.
Use the suite's classifier settings for the StatNN protocol. StatNN is not part
of the layer-wise sweep.

Each architecture shares one cached feature bank across its seeds. Separate
architectures may be run in parallel with `--models`; do not launch concurrent
writers for the same architecture/output directory. The runner saves the
classifier and predictions for every seed automatically (`--save-weights` is
not needed). `best_iteration` is a zero-based boosting iteration; for compatible
aggregation, the generic `best_epoch` column contains this same index, not a
neural-training epoch. Existing selected-layer StatNN results are a different
method and must not be reused as full-model results.

## Layers used by each experiment

StatNN is full-model and has no layer ID. For the layer-based methods,
layer IDs are the zero-based repository indices in `dsprobe/assets/layers`.
A table does not have to probe one layer. Each *method* is evaluated at the
layers validation selected for it, so two rows of the same table can sit at
different layers.

The three four-branch methods each keep one set throughout, for ResNet,
SupViT, MAE and DINO:

- **ProbeX (x4): 98, 59, 64, 59**
- **MVProbe: 55, 59, 64, 47**
- **R-DSProbe, and every variant that only changes its reference: 98, 53, 65, 47**

Tables 2, 3, 11 and 13 mix rows from these sets for exactly that reason.

Three experiments depart from it, each because the comparison forces one layer
on every condition:

- **Tables 6 and 10** change the branch composition rather than the reference, so
  every condition, displacement variants included, is held at the MVProbe set.
- **Table 4** trains one encoder over four trees, so a tree has one layer for all
  four conditions; it uses the MVProbe set throughout. Probing ResNet at L98
  would make its reference alone eight times larger and would dominate the
  parameter comparison the table is about.
- **Figure 2** matches the layer *size* across methods; see the note below.
- **Tables 7 and 8** are an accounting of shapes rather than a measurement, so they
  are reported at shapes that make the overhead comparable; ResNet sits at L55 for
  the same reason as in Figure 2.

| Paper item | Command | ResNet | SupViT | MAE | DINO |
|---|---|---|---|---|---|
| Table 2 — StatNN | `--suite configs/statnn.json` | full_model | full_model | full_model | full_model |
| Table 2 — ProbeX (×4) | `--suite configs/modelj_fixed.json` | 98 | 59 | 64 | 59 |
| Table 2 — MVProbe | (same suite, `layer_overrides`) | 55 | 59 | 64 | 47 |
| Table 2 — DSProbe, R-DSProbe | (same suite, default `layers`) | 98 | 53 | 65 | 47 |
| Table 2 — SD-LoRA (SD-200, SD-1k) | `--suite configs/sd_lora.json` | 46 | 46 | 46 | 46 |
| Table 4 — cross-tree (all four conditions) | `python -m dsprobe.shared` | 55 | 59 | 64 | 47 |
| Tables 7, 8 — parameter and FLOP accounting | `python -m dsprobe costs` | 55 | 59 | 65 | 47 |
| Table 6 — higher-order | `--suite configs/higher_order.json` | 55 | 59 | 64 | 47 |
| Table 10 — single-branch | `--suite configs/single_branch.json` | 55 | 59 | 64 | 47 |
| Table 3 — reference choice (R-DSProbe variants) | `--suite configs/reference_choice.json` | 98 | 53 | 65 | 47 |
| Table 3 — reference choice (MVProbe row) | (same suite, `layer_overrides`) | 55 | 59 | 64 | 47 |
| Table 11 — free offset (offset variants, R-DSProbe) | `python -m dsprobe.offset` | 98 | 53 | 65 | 47 |
| Table 11 — free offset (MVProbe row) | `--suite configs/modelj_fixed.json --methods mvprobe` | 55 | 59 | 64 | 47 |
| Table 13 — reference init (R-DSProbe variants) | `--suite configs/reference_init.json` | 98 | 53 | 65 | 47 |
| Table 13 — reference init (MVProbe column) | (same suite, `layer_overrides`) | 55 | 59 | 64 | 47 |
| Table 15 — update cap | `--suite configs/modelj_fixed.json --methods r_dsprobe --cap V` | 98 | 53 | 65 | 47 |
| Figures 1 and 3, Table 12 | `--suite configs/modelj_layerwise.json` | every matrix | every matrix | every matrix | every matrix |
| Figure 2, Table 9 | `python -m dsprobe.probe_sweep` (`configs/probe_sweep.json`) | 55 | 53 (ours) / 59 (MVProbe) | 65 (ours) / 64 (MVProbe) | 47 |

**Table 2** evaluates each layer-based method at its own validation-selected
layer; StatNN instead uses all named parameters. The layer-based rows differ
because those layers are what Figure 1's sweep selects. Table 4
is the opposite case: all four of its conditions --- the raw and referenced
per-tree experts and both shared-encoder variants --- are measured at the same layers
(`configs/shared.json`), because the table compares parameter counts and a
per-tree expert has to be counted at the layer the shared model also probes.

**Figure 2 and Table 9** are the exception in the other direction. The
probe-count comparison reports FLOPs at a given number of probes, so both
methods have to probe a layer of the same size. For SupViT, MAE and DINO the
two validation-selected layers happen to share a shape ($768\times768$); for
ResNet they do not (ours L98, MVProbe L55), so both methods are evaluated at
L55 — the unfavorable choice for our method, since it is not its own best layer.

**Tables 3 and 13 follow Table 2's convention rather than fixing one layer.**
Both tables exist to explain R-DSProbe: every row but one is R-DSProbe with a
different reference or a different starting point for that reference, so those
rows are probed where R-DSProbe is probed (98, 53, 65, 47). The raw-weight row
is MVProbe, a different method with its own validation-selected layers
(55, 59, 64, 47), exactly as in Table 2. Holding all rows at one layer would put
MVProbe somewhere it was not selected for and would understate it.

**MAE appears as 64 and 65** for the same reason: 64 is MVProbe's selected layer
and 65 is ours. The branch-composition ablations (Tables 6 and 10) vary the
architecture rather than the reference, and hold every condition at the MVProbe
layer set.

## Data and reference preparation

Use the released [Model Jungle dataset](https://huggingface.co/datasets/ProbeX/Model-J)
and preserve its original train/validation/test assignments. Dataset access and
redistribution remain subject to the original providers' terms.

```text
data/
  ResNet/                         # also SupViT, MAE, DINO
    train/                        # also val, test
      models--<owner>--<model>/
        refs/main                 # snapshot revision
        snapshots/<revision>/model.safetensors
  SD_200/                         # also SD_1k
    train/                        # also val, test, val_holdout, test_holdout
      <model-directory>/pytorch_lora_weights.safetensors
references/
  ResNet.bin
  SupViT.safetensors
  MAE.safetensors
  DINO.bin
```

Existing local banks can be used directly through `--data-root`; they need not
be copied. The downloader is opt-in and can fetch either or both kinds of data:

```bash
python -m dsprobe.download --subset ResNet SupViT MAE DINO --references --dry-run
python -m dsprobe.download --subset ResNet SupViT MAE DINO \
  --data-root data --reference-root references --references --allow-download
python -m dsprobe.download --subset SD_200 SD_1k --data-root data --allow-download
# References only:
python -m dsprobe.download --references --reference-models ResNet SupViT MAE DINO \
  --data-root data --reference-root references --allow-download
```

**Full banks can require more than a terabyte.** Check storage before downloading.
The downloader fetches weights, not training images. Caches are placed under
`data/.cache`; relative links avoid duplicate checkpoint copies. Keep that cache
with the dataset when transferring it. Missing models cause a failure report,
not a silently changed experimental split. Training performs no network access.

Discriminative safetensors need `dataset_chosen_targets` metadata containing 50
CIFAR-100 class names; targets are 100-bit membership vectors. Layer IDs are the
zero-based repository indices in `dsprobe/assets/layers`, not forward positions.
The historical ResNet convolution matricization is preserved.

Pretrained references are the original backbones before CIFAR-100 fine-tuning:

| Model | Upstream repository | Pinned revision | Local filename |
|---|---|---|---|
| ResNet | `microsoft/resnet-101` | `c2bf50f68263a35f102eb5c84ca91fc7352ceff3` | `ResNet.bin` |
| SupViT | `google/vit-base-patch16-224` | `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` | `SupViT.safetensors` |
| MAE | `facebook/vit-mae-base` | `25b184bea5538bf5c4c852c79d221195fdd2778d` | `MAE.safetensors` |
| DINO | `facebook/dino-vitb16` | `f205d5d8e640a89a2b8ef0369670dfc37cc07fc2` | `DINO.bin` |

Alternatively, provide `references/MODEL/layer_N.safetensors` with a matrix named
`reference`, already matricized to match the selected bank layer. Optional
`model` and `modelj_key` metadata are validated. PyTorch state dictionaries use
`weights_only=True`. R-DSProbe initializes G from the **training-bank mean** and
does not require pretrained-reference files.

SD-LoRA files need `imagenet_class_id` metadata and paired `.lora.down.weight`
and `.lora.up.weight` tensors. Layer IDs follow the sorted factor pairs in the
first training checkpoint. The input matrix is reconstructed as `B @ A`.
No images or Stable Diffusion image generation are needed.

Prepare frozen CLIP ViT-L/14 text features once per subset:

```bash
python -m dsprobe prepare-text --data-root data --subset SD_200 \
  --output references/text --clip-cache data/clip --device cuda:0 --allow-download
python -m dsprobe prepare-text --data-root data --subset SD_1k \
  --output references/text --clip-cache data/clip --device cuda:0 --allow-download
```

For offline preparation, place `ViT-L-14.pt` in `data/clip` and omit
`--allow-download`. Training validates the saved class ordering. Prompts are
`a photo of a {class_name}`; the ImageNet class-name mapping is bundled.

## 1. Single-layer fixed

```bash
python -m dsprobe train --suite configs/modelj_fixed.json --dry-run
python -m dsprobe train --suite configs/modelj_fixed.json \
  --data-root data --reference-root references --output results/fixed --device cuda:0
```

| Method | ResNet | SupViT | MAE | DINO |
|---|---:|---:|---:|---:|
| ProbeX (`probex4`) | 98 | 59 | 64 | 59 |
| MVProbe (`mvprobe`) | 55 | 59 | 64 | 47 |
| DSProbe (`dsprobe`) | 98 | 53 | 65 | 47 |
| R-DSProbe (`r_dsprobe`) | 98 | 53 | 65 | 47 |

Override layers independently per architecture for the selected methods:

```bash
python -m dsprobe train --suite configs/modelj_fixed.json \
  --methods dsprobe,r_dsprobe --model-layers ResNet=98 SupViT=59 MAE=65 DINO=47 \
  --k 128 --proj-dim 128 --rep-dim 512 \
  --data-root data --reference-root references --output results/fixed_custom --device cuda:0
```

Use a separate invocation for another method-specific override, or edit
`layer_overrides` in the JSON suite. `--models SupViT --layers 53` is the simpler
single-model form. `--layers` and `--model-layers` cannot be combined.

Common arguments:

- `--k` / `--n-probes`: base K, default **128** per branch. `probex4` uses one
  branch with **4K = 512** probes. This matches total probe count, not every
  parameter introduced by a learned reference.
- `--proj-dim`: projected features **per probe**, default **128**.
- `--rep-dim`: fused representation dimension, default **512**.
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--eval-every`: explicit
  training overrides, recorded with the results.
- `--cap 0.001`: relative reference step cap; `--cap 0` disables it.

## 2. SD-LoRA ID / zero-shot

```bash
python -m dsprobe train --suite configs/sd_lora.json --dry-run
python -m dsprobe train --suite configs/sd_lora.json \
  --data-root data --text-root references/text --output results/sd --device cuda:0
```

Defaults: SD-200 and SD-1k, **L46**, ProbeX **K=512** and four-view methods
**K=128 per branch**. All three use **500 epochs**, Adam `3e-4`, weight decay
`1e-5`, batch 128, validation at epochs 1, 11, ... and the final epoch.
An epoch is a full pass through the training models, not one optimizer update.
Layers and hyperparameters can be overridden as in experiment 1.

Frozen 768-dimensional CLIP text features receive a learned projection into
the probe representation space. Cosine similarities in that learned space are
the logits, trained with **cross-entropy** on training classes only.

| Subset | ID candidate classes | Zero-shot candidate classes |
|---|---:|---:|
| SD-200 | 170 | 30 |
| SD-1k | 850 | 150 |

ID and zero-shot each select their own state using the corresponding validation
group; ties retain the later state. Tests are evaluated only after selection.
This is candidate-group classification, **not combined 200/1000-way evaluation**.
The main SD suite includes ProbeX, MVProbe, and R-DSProbe. Pretrained DSProbe
is not included because `B @ A` is already a LoRA update and supplies no
pretrained weight matrix in matching coordinates.

## 3. Cross-tree shared encoder

```bash
python -m dsprobe.shared --config configs/shared.json --dry-run
python -m dsprobe.shared --config configs/shared.json \
  --data-root data --output results/shared --device cuda:0
python -m dsprobe.shared --config configs/shared.json --output results/shared --summarize
```

Four conditions forming a 2x2 over {raw, learned reference} x {per-tree,
shared}: raw per-tree specialists (`specialist_raw`), per-tree specialists with
their own learned references (`specialist`), a shared raw-response encoder
(`shared_raw`), and a shared refined-response encoder (`shared_ref`). Defaults
use all four trees at **55/59/64/47**.

All four are fitted by this runner, not taken from experiment 1. The per-tree
rows would otherwise mix two training harnesses: `dsprobe train` initializes
probes with Xavier, orders minibatches by the repository rule, and validates at
epochs 1, 26, ...; this runner uses PyTorch's default linear initialization, a
shuffled permutation, and validates at epochs 25, 50, .... Those differences are
small but they are not the reference, so a row copied across runners would not
isolate what the table varies.
Each tree retains its own probes, projections, and (when refined) reference;
there is **one common fusion encoder and classification head**. There is not
one universal reference matrix across differently shaped trees.

Defaults: 32 examples **per tree** per update, sum of tree-wise BCE losses,
500 epochs, Adam `3e-4`, and whole-model weight decay `1e-5`.
The shared state is selected by macro validation accuracy; specialists are
selected individually. Validation occurs every 25 epochs and at the final epoch.
See `--help` for model, layer, seed, probe-count, and dimension overrides.

## 4. MVProbe higher-order ablation

```bash
python -m dsprobe train --suite configs/higher_order.json \
  --data-root data --output results/higher_order --device cuda:0
```

All conditions use **55/59/64/47**, with jointly trained standardized branches:

| Method | Branch composition |
|---|---|
| `mvprobe` | Four first-/second-order views |
| `poly6` | Four views plus left/right degree 3 |
| `duplicate6` | Four views plus independent first-order duplicates |
| `poly8` | Four views plus left/right degrees 3 and 4 |
| `duplicate8` | Four views plus independent first-/second-order duplicates |

Every branch uses K=128 by default. Each six/eight-branch control matches the
parameter count of the corresponding higher-order model. Matrix products are
evaluated associatively without materializing dense square Gram matrices.

## 5. Single-branch responses

```bash
python -m dsprobe train --suite configs/single_branch.json \
  --data-root data --reference-root references --output results/single --device cuda:0
```

All nine conditions use **55/59/64/47**. Each is a separately trained single-branch
network, **not a branch masked out of a jointly trained four-branch model**.
All nine use **200 epochs, Adam with learning rate `2e-3`, and weight decay `0`**.

| Response | Raw | Frozen pretrained G | Learned G |
|---|---|---|---|
| First order | `linear_raw` | `linear_pretrained` | `linear_learned` |
| Left Gram | `gram_left_raw` | `gram_left_pretrained` | `gram_left_learned` |
| Right Gram | `gram_right_raw` | `gram_right_pretrained` | `gram_right_learned` |

Centered responses are `(W-G)U`, `(WW^T-GG^T)U`, and `(W^T W-G^T G)U`.
Gram subtraction is **not** the Gram of `W-G`. Each learned reference starts
from the training mean and is trained independently for its single branch.

The nine conditions share the epoch count, optimizer, learning rate and weight
decay, but not every other setting: first-order conditions use the classic
unstandardized ProbeX protocol (default probe initialization, the legacy
shuffle, no final-epoch evaluation) while Gram conditions use response
standardization, Xavier probe initialization, the repository shuffle and a
final-epoch evaluation. The three conditions **within** a response type are
therefore identical except for the input, which is the comparison the table
makes; a number is not directly comparable **across** response types.

## 6. Layer-wise sweep

```bash
python -m dsprobe train --suite configs/modelj_layerwise.json \
  --data-root data --reference-root references --output results/layerwise --device cuda:0
```

This independently fits ProbeX, MVProbe, DSProbe, and R-DSProbe per matrix.
**The full sweep is 6,520 runs** (105 ResNet matrices, 74 per ViT, four methods,
five seeds), which is GPU-days rather than GPU-hours. Check the count with
`--dry-run`, then shard the work with `--models`, `--methods`, `--layers` and
`--seeds` as in the example below.
ProbeX in this sweep uses K=128, 200 epochs, learning rate `2e-3`, and weight
decay `0`. ResNet has 105 indexed matrices and each ViT
has 74. DSProbe excludes classifier layer 0, whose pretrained head is incompatible.
Filter and shard jobs explicitly, for example:

```bash
python -m dsprobe train --suite configs/modelj_layerwise.json \
  --models SupViT --methods dsprobe --layers 50,51,52,53 --seeds 1,2,3,4,5 \
  --data-root data --reference-root references --output results/layerwise --device cuda:0
```

Layer selection uses **mean validation accuracy**, not test accuracy. Missing
layers remain gaps. Optional analysis uses the bundled forward-order maps:

```bash
python -m dsprobe analyze --results results/layerwise --output results/layerwise_summary --layerwise
```

## 7. Free-offset control

```bash
python -m dsprobe.offset --config configs/offset.json --dry-run
python -m dsprobe.offset --config configs/offset.json \
  --data-root data --output results/offset --device cuda:0
python -m dsprobe.offset --config configs/offset.json --output results/offset --summarize
```

Defaults use **98/53/65/47**, R-DSProbe's layers, because every condition here
is R-DSProbe with a different offset parameterization. The three conditions are
free response offsets initialized at zero, free response offsets initialized to
the negative training-mean reference responses, and a learned weight reference.
Offsets are added **before response standardization**.

The raw-weight (MVProbe) row of the table comes from
`configs/modelj_fixed.json`, at MVProbe's own layers (55/59/64/47), for the same
reason it does in Table 2. Pass `--variants none` to this runner instead if you
want the no-offset condition at R-DSProbe's layers.

For training mean M, initial offsets are `-MU`, `-M^T U`, `-M(M^T U)`, and
`-M^T(MU)`, using the respective initial probes. Free offsets then train
independently of the probes. Learned-reference subtraction is recomputed from
the current G and probes. Defaults are
500 epochs, Adam `3e-4`, weight decay `1e-5`, validation every 25 epochs and at
the final epoch.

## Architecture and training details

For a response `[batch, response_dim, K]`, a bias-free projection maps each
probe to `proj_dim` features, followed by ReLU. Each branch is flattened to
`K * proj_dim`. Four-branch concatenation has **65,536 features**, at default
K=128 and projection dimension 128, then a learned linear encoder maps it to
**512**, followed by the prediction head. ProbeX uses the same per-probe
projection/flattening pattern with a single branch.

Model-J uses BCE with logits. Four-view methods, higher-order controls, and
ProbeX (x4) use Adam `3e-4`, weight decay `1e-5`, and 500 epochs. Ordinary ProbeX
and all nine single-response variants use Adam `2e-3`, weight decay `0`, and
200 epochs. The batch size is 128 except for the per-tree batching specified
for shared training. Where enabled, weight decay applies to all trainable
parameters. Standardized responses subtract their per-sample branch mean
and divide by sample standard deviation plus `1e-8`. No dataset-wide response
statistics are learned.

The main Model-J runners validate every 25 epochs starting at epoch 1.
Standardized methods also evaluate the final epoch: 500 for the four-/higher-view
methods and 200 for the single Gram branches. Classic ProbeX and first-order
single branches validate at epochs 1, 26, ..., 176; ProbeX (x4) validates at
1, 26, ..., 476. These classic runs use their deterministic shuffle rule and
do not add a final-epoch evaluation. Ties retain the earlier state. Shared and
offset use the separately stated 25-epoch cadence.

The optional cap bounds **each post-Adam reference update** by
`cap * ||G_initial||_F`, with a fixed initial norm. It does not bound cumulative
drift, add an anchor penalty, or reset optimizer moments. Main R-DSProbe uses
cap 0.001. Frozen references are not optimizer parameters.

Gram differences are evaluated as written, `X(X^T U) - G(G^T U)`, never as the
Gram of `X - G`. The reference term does not depend on the batch, so it is
computed once per forward. Its six matrix products add `12*m*n*r/B` forward
FLOPs per example at batch size B. At B=128 and the accounting shapes in
Tables 7 and 8, the forward matrix-multiplication count grows by less than 0.7%
over MVProbe. Reference-inclusive counts are used throughout the FLOP reports.

## 8. Reference choice (Table 3)

```bash
python -m dsprobe train --suite configs/reference_choice.json --dry-run
python -m dsprobe train --suite configs/reference_choice.json \
  --data-root data --reference-root references --output results/reference --device cuda:0
python -m dsprobe analyze --results results/reference --output results/reference_summary
```

Six four-branch conditions differing only in what is subtracted before probing.
`configs/reference_choice.json` selects five of them; the two random variants
are alternatives to each other, as explained below. The suite probes the
R-DSProbe layers and overrides only `mvprobe` to its own, so the raw-weight
baseline is not evaluated at a layer chosen for a different method.

| Method | Reference |
|---|---|
| `mvprobe` | none; the raw weight |
| `dsprobe_rand` | a random matrix as drawn |
| `dsprobe_rand_norm` | a random matrix rescaled to the pretrained reference's norm |
| `dsprobe` | the frozen pretrained origin |
| `dsprobe_mean` | the frozen training-bank mean, i.e. centering |
| `r_dsprobe` | the learned reference |

The random references are drawn from a generator seeded by model, layer and
seed, so each seed uses its own matrix and the run stays reproducible.
`dsprobe_rand_norm` reads the pretrained reference to match its norm and
therefore needs `--reference-root`; `dsprobe_rand` and `dsprobe_mean` do not.

The two random variants answer different questions and are not interchangeable.
An unscaled draw is far larger than the weight it is subtracted from, so
`dsprobe_rand` changes the size of the input as well as its location, whereas
`dsprobe_rand_norm` holds the size fixed and moves only the location --- which
is what isolates *where* the reference sits from *how much* is removed.
`configs/reference_choice.json` uses `dsprobe_rand_norm`; swap in `dsprobe_rand`
if the unscaled draw is the intended control. State in the table which one the
reported row used, since "a random matrix of the same shape" describes both.

`dsprobe_mean` is `r_dsprobe` with its reference frozen at the initial value,
which is what separates centering from learning the reference.

## 9. Reference initialization (Table 13)

```bash
# MVProbe baseline: run once.
python -m dsprobe train --suite configs/reference_init.json \
  --methods mvprobe --cap 0 \
  --data-root data \
  --output results/table13/mvprobe --device cuda:0

# W_mean, W_pre, W_rand, and W_rand + Norm-matched: uncapped.
for INIT in mean pretrained random random_norm; do
  python -m dsprobe train --suite configs/reference_init.json \
    --methods r_dsprobe --learned-init "$INIT" --cap 0 \
    --data-root data --reference-root references \
    --output "results/table13/init_${INIT}_uncapped" --device cuda:0
done

# W_mean + Cap (ours).
python -m dsprobe train --suite configs/reference_init.json \
  --methods r_dsprobe --learned-init mean --cap 1e-3 \
  --data-root data --reference-root references \
  --output results/table13/init_mean_capped --device cuda:0
```

Table 13 compares four reference initializations without the reference-update
cap (`--cap 0`), followed by mean initialization with the cap enabled
(`--cap 1e-3`). The CLI defaults to `--cap 1e-3`, so uncapped runs must
explicitly specify `--cap 0`. The `pretrained` and `random_norm` initializations
read the pretrained reference file and require `--reference-root`. The
initialization choice is recorded in each result's contract as
`learned_reference_init`.

The MVProbe baseline is run once using its method-specific validation-selected
layers. Results are saved separately for each condition; analyze each condition
directory independently.

## Outputs, resume, and verification

Main runners write one `result.json` per model/layer/method/seed with metrics,
validation history, configuration, parameter count, and code/data fingerprints.
No model weights are saved unless `--save-weights` is supplied. No reconstructed
model-bank cache is written. `--resume` skips completed jobs only when their
contracts match; it does not restart an interrupted optimizer trajectory.

```bash
python -m dsprobe analyze --results results/fixed --output results/fixed_summary
python -m dsprobe analyze --results results/sd --output results/sd_summary
python -m dsprobe analyze --results results/higher_order --output results/higher_order_summary
python -m dsprobe analyze --results results/single --output results/single_summary
```

Aggregates report mean and **population** standard deviation across actual
completed seeds. Incomplete groups are marked. Shared and offset have their
own `--summarize` commands above; repeat any custom training overrides when
summarizing those runs. Use a new output directory when changing a
protocol. Training itself does not generate figures or tables.

Random-reference runs retain each seed's initial-reference fingerprint, while
aggregation compares their common initialization settings. Norm-matched runs
also record the pretrained reference used as the norm target. Older results
without that target fingerprint retain the original initial-reference check.

CPU tests use temporary synthetic inputs outside the source directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Tests do not download data or fit full banks. Full reproduction requires the
public data and adequate CPU RAM / accelerator memory. Inspect `--dry-run`
before launching and use model/method/layer/seed filters to distribute jobs.
