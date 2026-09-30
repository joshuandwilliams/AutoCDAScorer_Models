# AutoCDAScorer_Models

[![Tests](https://github.com/joshuandwilliams/AutoCDAScorer_Models/actions/workflows/test.yml/badge.svg)](https://github.com/joshuandwilliams/AutoCDAScorer_Models/actions/workflows/test.yml)
[![Lint](https://github.com/joshuandwilliams/AutoCDAScorer_Models/actions/workflows/lint.yml/badge.svg)](https://github.com/joshuandwilliams/AutoCDAScorer_Models/actions/workflows/lint.yml)
![python](https://img.shields.io/badge/python-3.12-1e90ff?logo=python) ![tensorflow](https://img.shields.io/badge/tensorflow-2.21.0-orange) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Training and analysis code behind the AutoCDAScorer and CDALabeller R packages. Builds the CNN/ViT models that score agroinfiltration cell death severity (0–6) from cropped leaf images, and runs the analyses comparing them against hand-crafted features, human raters, and each other.

## Status

See [`RERUN.md`](RERUN.md) for an open rerun: the geometric- and cGAN-augmented grids (`models/07`, `models/08`) cross-validated over a differently pooled dataset than the unaugmented grids (`models/03`–`05`), which confounds the augmentation comparison until the affected results are regenerated.

## What's here

| Path | Contents |
|------|----------|
| `src/` | Dataset construction, CNN/ViT model builders, augmentation (geometric and cGAN), random hyperparameter search, and results storage/metrics |
| `models/` | One folder per model family (base/medoid CNN and ViT, soft labels, cGAN, geometric and cGAN augmentation), each holding its search configs and results |
| `analyses/` | Quarto analyses numbered to match Chapter 3's Results section — see [`analyses/ORDER.md`](analyses/ORDER.md) |
| `data/` | Cropped images, combined score data, and the project anonymisation key |
| `containers/` | Singularity definition for the HPC TensorFlow GPU image |
| `scripts/` | HPC sync helpers |

## Environment

```bash
conda env create -f environment.yml
conda activate agroinfiltration_scoring
```

The pinned HPC training environment mirrors `containers/TensorFlowGPU_2_21_0.def`; `analyses/environment.yml` is a separate, single environment for rendering all analyses.

## Tests

```bash
pip install -e '.[test]'
pytest
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
