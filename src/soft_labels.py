"""Build soft training targets from the three scorers' scores.

Every CDA in ``data/combined_cda_data_median.csv`` carries three independent human
scores. Every search so far has trained on a one-hot of their *median*, which throws away
the disagreement: only 976 of the 3,052 dataset crops have all three scorers agreeing, so
roughly two thirds of the training signal has uncertainty in it that the one-hot target
asserts does not exist.

A soft target keeps it. For a CDA scored 3, 4, 4 the target is
``[0, 0, 0, 1/3, 2/3, 0, 0]`` rather than ``[0, 0, 0, 0, 1, 0, 0]``. Cross-entropy against
that distribution penalises confident predictions of 3 less than confident predictions of
0, which is the ordering the scale implies and the one-hot target cannot express.

This is NOT label smoothing. Smoothing spreads a fixed epsilon uniformly over classes the
raters never mentioned; this puts mass exactly where the raters put it, so the amount and
the direction of the softening are measured rather than chosen.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SCORE_COLS = ["Score1", "Score2", "Score3"]


def _key(basename: pd.Series, row, col, pos) -> pd.Series:
    """Join key matching a crop filename: ``<Basename without ext>_<Row>_<Col>_<Pos>``."""
    stem = basename.astype(str).str.replace(r"\.[Tt][Ii][Ff]$", "", regex=True)
    return (
        stem + "_" + row.astype(str) + "_" + col.astype(str) + "_" + pos.astype(str)
    ).str.lower()


def build_soft_targets(
    split_csv: str | Path,
    rater_csv: str | Path,
    num_classes: int = 7,
    split: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (soft_targets, merged) aligned row-for-row with ``split_csv``.

    Parameters
    ----------
    split_csv
        A dataset manifest written by ``base_dataset.save_dataset`` (columns
        ``split, filename, full_path, label``). Row order defines the output order.
    rater_csv
        ``data/combined_cda_data_median.csv``, one row per CDA with three scorer columns.
    split
        If given, keep only that split (e.g. ``"train"``). Order is preserved.

    Raises if any crop fails to match a rater record, or if the manifest's ``label``
    disagrees with the raters' ``Median_Score`` -- either would mean the targets are
    silently misaligned with the images.
    """
    manifest = pd.read_csv(split_csv)
    if split is not None:
        manifest = manifest[manifest["split"] == split].reset_index(drop=True)
    raters = pd.read_csv(rater_csv)

    manifest = manifest.copy()
    manifest["_key"] = (
        manifest["filename"].astype(str).str.replace(r"\.tif$", "", regex=True).str.lower()
    )
    raters = raters.copy()
    raters["_key"] = _key(raters["Basename"], raters["Row"], raters["Col"], raters["Pos"])

    merged = manifest.merge(raters, on="_key", how="left", validate="one_to_one")
    missing = merged["Median_Score"].isna().sum()
    if missing:
        raise ValueError(f"{missing} crops have no rater record; targets would be misaligned")
    if not (merged["label"] == merged["Median_Score"]).all():
        raise ValueError("manifest label != rater Median_Score; the join is wrong")

    # 148 of the 6,364 CDAs were scored by only two of the three raters (one NaN each).
    # The distribution is built from the raters who actually scored the CDA and normalised
    # by how many that was, so a 2-rater CDA scored 0, 0 gives [1, 0, ...] rather than
    # [2/3, ...]. Treating a missing score as a vote for class 0 would be a silent bias
    # towards the most common class.
    scores = merged[SCORE_COLS].to_numpy(dtype="float64")
    targets = np.zeros((len(merged), num_classes), dtype="float32")
    counts = np.zeros(len(merged), dtype="float32")
    rows = np.arange(len(merged))
    for j in range(scores.shape[1]):
        ok = ~np.isnan(scores[:, j])
        np.add.at(targets, (rows[ok], scores[ok, j].astype(int)), 1.0)
        counts += ok
    if (counts == 0).any():
        raise ValueError("some CDAs have no scorer values at all")
    targets /= counts[:, None]

    n_partial = int((counts < scores.shape[1]).sum())
    if n_partial:
        print(
            f"note: {n_partial} CDAs were scored by fewer than {scores.shape[1]} raters; "
            f"their targets are normalised over the raters present"
        )
    return targets, merged


def summarise(targets: np.ndarray) -> dict:
    """Agreement structure of a soft-target matrix, for logging."""
    peak = targets.max(axis=1)
    return {
        "n": len(targets),
        "unanimous (all 3 agree)": int((peak >= 1.0 - 1e-6).sum()),
        "two of three agree": int(((peak > 0.5) & (peak < 1.0 - 1e-6)).sum()),
        "all three differ": int((peak <= 0.5 + 1e-6).sum()),
        "mean peak mass": float(peak.mean()),
        "mean entropy (nats)": float(
            -(targets * np.log(np.clip(targets, 1e-12, None))).sum(1).mean()
        ),
    }
