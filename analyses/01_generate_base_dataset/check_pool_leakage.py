"""Fail if an augmentation pool contains any ensemble or test crop.

`base_dataset.load_images_and_labels` lists the crop folders in filesystem order, so the
seeded split depends on the machine it runs on. A pool built on the cluster in September
used a different split from `medoid_64`, and 227 of its source crops were test crops. Run
this on every new pool before training on it.

    python analyses/01_generate_base_dataset/check_pool_leakage.py \\
        data/datasets/medoid_64_aug_trainval/medoid_64_aug.npz
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "src"))

import base_dataset  # noqa: E402


def held_out_names() -> dict[str, set[str]]:
    ds = base_dataset.load_dataset(str(REPO / "data/datasets/medoid_64/medoid_64.npy"))
    return {
        split: {Path(str(f)).name for f in np.asarray(ds[f"{split}_filepaths"]).tolist()}
        for split in ("ensemble", "test")
    }


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit("usage: check_pool_leakage.py POOL.npz [POOL.npz ...]")
    held = held_out_names()
    failed = False
    for path in paths:
        names = {Path(str(n)).name for n in np.load(path, allow_pickle=True)["source_names"]}
        overlap = {split: len(names & held[split]) for split in held}
        print(f"{path}: {len(names)} source crops, overlap {overlap}")
        failed |= any(overlap.values())
    if failed:
        sys.exit("LEAK: a pool contains ensemble or test crops. Do not train on it.")
    print("No ensemble or test crops in any pool.")


if __name__ == "__main__":
    main(sys.argv[1:])
