# hloc — deep image matching & SfM pipeline

[hloc (Hierarchical-Localization)](https://github.com/cvg/Hierarchical-Localization) is a Python framework by ETH CVG that wraps SuperPoint, DISK, ALIKED, LightGlue, LoFTR, NetVLAD, and vocabulary-tree retrieval around a COLMAP SfM / localization backend — all under one interface.

## Install

The `OpenCVProjects` conda env (`environment.yml`) already pulls in `colmap`, `pycolmap`, `pytorch`, `torchvision`, `kornia`, `einops`, `h5py`. To add hloc itself (it ships its extractors as git submodules, so `--recursive` is required):

```bash
conda activate OpenCVProjects
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git thirdparty/hloc
python -m pip install -e thirdparty/hloc
```

## Built-in configs

After install, the catalog of extractor / matcher / retrieval configs is:

```python
from hloc import extract_features, match_features
list(extract_features.confs.keys())
#  → ['superpoint_aachen', 'superpoint_max', 'superpoint_inloc',
#     'r2d2', 'd2net-ss', 'sift', 'sosnet',
#     'disk', 'aliked-n16',
#     'dir', 'netvlad', 'openibl', 'megaloc']

list(match_features.confs.keys())
#  → ['superpoint+lightglue', 'disk+lightglue', 'aliked+lightglue',
#     'superglue', 'superglue-fast',
#     'NN-superpoint', 'NN-ratio', 'NN-mutual', 'adalam']
```

Vocabulary-tree retrieval is available via the underlying `pycolmap` backend (`pycolmap.VocabTree`).

## When to use hloc vs. kornia

- **hloc** — full end-to-end pipeline (retrieval → pairs → extraction → matching → COLMAP reconstruction or localization). Output is a COLMAP model.
- **kornia** (already in the env) — native PyTorch modules (`kornia.feature.SuperPoint`, `LightGlue`, `LoFTR`, `DISK`) for embedding into your own code. No pipeline glue.

## Minimal end-to-end pipeline

```python
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction

images = Path("data/my_scene")
outputs = Path("outputs/my_scene")

retrieval_conf = extract_features.confs["netvlad"]
feature_conf   = extract_features.confs["superpoint_aachen"]
matcher_conf   = match_features.confs["superpoint+lightglue"]

# 1. Global descriptors → image pairs
retrieval = extract_features.main(retrieval_conf, images, outputs)
pairs = outputs / "pairs-netvlad.txt"
pairs_from_retrieval.main(retrieval, pairs, num_matched=20)

# 2. Local features + matching
features = extract_features.main(feature_conf, images, outputs)
matches  = match_features.main(matcher_conf, pairs, feature_conf["output"], outputs)

# 3. COLMAP reconstruction
reconstruction.main(outputs / "sfm", images, pairs, features, matches)
```

Swap `superpoint_aachen` for `disk` or `aliked-n16`, and `superpoint+lightglue` for `disk+lightglue` / `aliked+lightglue` to try different feature stacks. For detector-free matching, use `match_dense` with the `loftr` config instead.
