# AutoCDAScorer_Models

[![Tests](https://github.com/joshuandwilliams/AutoCDAScorer_Models/actions/workflows/test.yml/badge.svg)](https://github.com/joshuandwilliams/AutoCDAScorer_Models/actions/workflows/test.yml)
[![Lint](https://github.com/joshuandwilliams/AutoCDAScorer_Models/actions/workflows/lint.yml/badge.svg)](https://github.com/joshuandwilliams/AutoCDAScorer_Models/actions/workflows/lint.yml)
[![Codecov test coverage](https://codecov.io/gh/joshuandwilliams/AutoCDAScorer_Models/graph/badge.svg)](https://app.codecov.io/gh/joshuandwilliams/AutoCDAScorer_Models)
![python](https://img.shields.io/badge/python-3.12-1e90ff?logo=python)
![tensorflow](https://img.shields.io/badge/tensorflow-2.21.0-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository trains and compares models that score cell death areas (CDAs)
in UV agroinfiltration images of *Nicotiana benthamiana* leaves. Each model
takes a cropped CDA image and gives it a cell death severity score from 0 to 6.

It holds two things. The first is the code that trains convolutional neural
networks (CNNs), vision transformers (ViTs) and conditional generative
adversarial networks (cGANs) on the cluster. The second is the analyses that
compare these models with hand-crafted features, with human raters and with
each other. The analyses are the results of Chapter 3 of the thesis.

The [AutoCDAScorer](https://github.com/joshuandwilliams/AutoCDAScorer) R
package uses 16 of the CNNs trained here. `analyses/package_export` builds and
checks the other files that the package ships.

## Status

The augmented grids (`models/07` and `models/08`) were cross-validated on the
training split alone. The unaugmented grids (`models/01` to `models/05`) were
cross-validated on the training and validation splits pooled together. The two
sets of results therefore differ in their real training images as well as in
augmentation. A rerun with matched pools is pending. Until then, a difference
between them is not a measure of the effect of augmentation alone.

## What is here

| Path | Contents |
|---|---|
| `src/` | Shared Python code: dataset building, model builders, augmentation, the cGAN, soft labels, the hyperparameter search and its results files |
| `models/` | One folder per model family, each with the scripts that run its hyperparameter search on the cluster |
| `analyses/` | Quarto analyses, numbered in the order of Chapter 3's results. See [`analyses/ORDER.md`](analyses/ORDER.md) |
| `data/` | The scores from all three raters, image information, the project anonymisation key and the scoring key |
| `containers/` | The Singularity recipe for the cluster's TensorFlow image |
| `scripts/` | Scripts that copy files to and from the cluster |

## Trained models

Each CNN and ViT hyperparameter search keeps only its 100 best models. These
are not stored in this repository because there are 600 of them. The 16 CNNs that
AutoCDAScorer uses are on Hugging Face at
[joshuandwilliams/AutoCDAScorer](https://huggingface.co/joshuandwilliams/AutoCDAScorer).
Eight cGAN generators are in `models/06_Medoid_CGAN/01_Conditional/`. One of
them generates the images for the cGAN-augmented grid.

## Environment

For training and tests:

```bash
conda env create -f environment.yml
conda activate agroinfiltration_scoring
```

This matches the cluster image, `containers/TensorFlowGPU_2_21_0.def`. The
analyses use a separate environment, `analyses/environment.yml`.

## Tests

```bash
pip install -e '.[test]'
pytest
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
