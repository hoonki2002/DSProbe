# Additional analyses

Run the commands from the repository root. Data and reference layout are described
in [REPRODUCE.md](REPRODUCE.md).

## Does R-DSProbe need the Model Tree labels?

R-DSProbe keeps one reference $W_\star$ per tree, initialized at the mean of that
tree's training leaves. It therefore needs the *tree membership* (which models
descend from the same root), not the full parent-child lineage and not the
pretrained model. These tools ask whether the membership can be recovered from the
weights alone, and what wrong membership would cost.

### Slim layer bank

The analyses read one or a few layers per model. `download_layers` fetches only the
selected tensors of each checkpoint with HTTP range requests and writes them in the
usual bank layout, which is a few MB per model instead of about 350 MB:

```bash
python -m dsprobe.download_layers --subset SupViT MAE DINO --layers 53 65 47 --dry-run
python -m dsprobe.download_layers --subset SupViT MAE DINO --layers 53 65 47 \
  --data-root data --allow-download
# Different layers per architecture
python -m dsprobe.download_layers --subset ResNet SupViT MAE DINO \
  --model-layers ResNet=55 SupViT=59 MAE=64 DINO=47 --data-root data --allow-download
```

The Hub limits anonymous requests; the downloader honors `Retry-After` and resumes
from what is already present. Lower `--workers` if you are rate limited.

### Recovering the trees from the weights

```bash
python -m dsprobe.tree_estimation distance --data-root data --reference-root references --output results/tree_estimation
python -m dsprobe.tree_estimation cluster  --data-root data --output results/tree_estimation
```

`cluster` pools the leaves of all trees without their labels, runs k-means on one
flattened layer, and compares the clusters with the true trees (adjusted Rand
index, purity, confusion matrix). It is run once on all splits and once on the
training leaves only, assigning the validation and test leaves to the nearest
centroid. `distance` reports the Frobenius distance from every leaf to every tree
root and to every tree mean, in absolute and relative terms; the separation of the
trees is judged on the absolute distance. Use `--layers` and `--seeds` to change
the probed layers and k-means seeds, and `all` to run both commands.

### Wrong tree membership

```bash
python -m dsprobe.contaminate --data-root data --output data_contam --fractions 0 5 10
python -m dsprobe train --suite configs/modelj_fixed.json --methods r_dsprobe \
  --data-root data_contam/p10 --reference-root references --output results/contam/p10
```

`contaminate` builds symlink banks in which a fraction $p$ of each tree's training
leaves is replaced by the leaves of the other trees nearest to that tree's training
mean (the most plausible confusions). Validation and test leaves stay clean, and
`manifest.json` records every replacement. The unchanged `train` command runs on
each bank.

## DSProbe in the cross-tree experiment

`dsprobe.shared` compares per-tree probes with a single encoder shared across trees.
Besides the raw-weight and R-DSProbe variants, `specialist_dsprobe` and
`shared_dsprobe` probe the displacement from the pretrained root, which is frozen
and therefore not learned:

```bash
python -m dsprobe.shared --data-root data_shared --reference-root references \
  --variants specialist_raw specialist_dsprobe specialist shared_raw shared_dsprobe shared_ref \
  --output results/shared --device cuda:0
python -m dsprobe.shared --output results/shared --summarize
```

`specialist` and `shared_ref` are R-DSProbe with one learned reference per tree.
The default layers are ResNet 55, SupViT 59, MAE 64, and DINO 47; `--layers` and
`--models` override them.
