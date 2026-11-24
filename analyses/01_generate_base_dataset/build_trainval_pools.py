"""Build the train+val augmentation pools into new folders.

Runs `build_medoid_64_aug.py` or `build_medoid_64_cgan.py` unchanged, except that their
output folder is redirected, so the current pools in `data/datasets/medoid_64_aug` and
`data/datasets/medoid_64_cgan` are not overwritten. Both builders pool the train and
validation splits by default (see RERUN.md).

    python analyses/01_generate_base_dataset/build_trainval_pools.py aug  --target 5000
    python analyses/01_generate_base_dataset/build_trainval_pools.py cgan --target 5000

The pools are written to `data/datasets/medoid_64_aug_trainval` and
`data/datasets/medoid_64_cgan_trainval`. Any other arguments are passed to the builder.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

BUILDERS = {
    "aug": ("build_medoid_64_aug.py", "medoid_64_aug_trainval"),
    "cgan": ("build_medoid_64_cgan.py", "medoid_64_cgan_trainval"),
}


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in BUILDERS:
        sys.exit("usage: build_trainval_pools.py {aug,cgan} [builder arguments]")
    if "--train-only" in argv:
        sys.exit("--train-only would rebuild the original pool. Run the builder directly.")
    script, folder = BUILDERS[argv[0]]
    spec = importlib.util.spec_from_file_location(Path(script).stem, HERE / script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.OUT = REPO / "data" / "datasets" / folder
    sys.argv = [script, *argv[1:]]
    builder.main()


if __name__ == "__main__":
    main(sys.argv[1:])
