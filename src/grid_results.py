"""Loading, deduplicating and tie-banding a hyperparameter grid's results CSV.

The CNN grid crosses ``reg = None`` with all three ``reg_strength`` values, but
``reg_strength`` has no effect when there is no regulariser, so three rows of the
8,640 describe the same model. Every CNN-family grid therefore holds 6,720
distinct settings, trained 8,640 times. Left alone this inflates configuration
counts, weights any mean over a tied set towards whichever settings happen to have
been trained three times, and lets one setting occupy three places in an ensemble.

``deduplicate`` keeps the best-scoring run of each setting; ``tie_band`` then takes
every setting within one standard error of the grid's best. Deduplicating first is
what makes a tied set a set of distinct settings.

The ViT grid needs none of this: it is built as two sub-grids so ``reg = None``
pairs with one strength only, and ``deduplicate`` is a no-op on it.
"""

import numpy as np
import pandas as pd

# Everything a grid CSV records about a trained model rather than about its settings.
METRIC_COLUMNS = frozenset(
    {
        "model_id",
        "avg_vaf",
        "avg_taf",
        "avg_divergence",
        "avg_val_nearmiss",
        "avg_val_qwk",
        "avg_vaf_orig",
        "avg_vaf_aug",
        "avg_taf_orig",
        "avg_taf_aug",
        "aug_minus_orig_vaf",
        "best_epoch",
        "train_seconds",
        "cv_seed",
        "test_acc",
        "test_nearmiss",
        "test_qwk",
        "task",
    }
)

# Crops pooled over the train and validation splits, the denominator of avg_vaf.
CV_CROPS = 2289


def load_grid(root) -> pd.DataFrame:
    """Main results CSV plus any un-merged per-task part files, one row per model_id."""
    frames = [pd.read_csv(root / "random_search_results.csv")]
    parts = sorted((root / "parts").glob("*.csv")) if (root / "parts").exists() else []
    frames += [pd.read_csv(p) for p in parts]
    return pd.concat(frames, ignore_index=True).drop_duplicates("model_id")


def setting_key(grid: pd.DataFrame) -> pd.Series:
    """One string per row identifying the model it describes.

    Every hyperparameter column takes part except ``reg_strength``, which is
    replaced by a constant wherever ``reg`` is absent, since it changes nothing
    there.
    """
    axes = [c for c in grid.columns if c not in METRIC_COLUMNS and c not in ("reg", "reg_strength")]
    reg = grid["reg"].fillna("none") if "reg" in grid.columns else pd.Series("none", grid.index)
    strength = (
        grid["reg_strength"].astype(str)
        if "reg_strength" in grid.columns
        else pd.Series("none", grid.index)
    )
    return (
        grid[axes].astype(str).agg("|".join, axis=1)
        + "|"
        + reg.astype(str)
        + "|"
        + np.where(reg == "none", "none", strength)
    )


def deduplicate(grid: pd.DataFrame, metric: str = "avg_vaf") -> pd.DataFrame:
    """Keep the highest-``metric`` run of each distinct setting.

    Ties on ``metric`` are broken by the lowest ``model_id``, so the choice does not
    depend on row order.
    """
    ordered = grid.sort_values([metric, "model_id"], ascending=[False, True])
    return ordered.loc[~setting_key(ordered).duplicated()].sort_index()


def standard_error(p: float, n: int = CV_CROPS) -> float:
    """Standard error of an accuracy ``p`` measured on ``n`` crops."""
    return float(np.sqrt(p * (1 - p) / n))


def tie_band(grid: pd.DataFrame, metric: str = "avg_vaf", n: int = CV_CROPS) -> pd.DataFrame:
    """Every row within one standard error of the best ``metric`` in the grid.

    Pass an already-deduplicated grid: on a raw one the band counts repeat runs of
    the same setting as separate members.
    """
    best = grid[metric].max()
    return grid[grid[metric] >= best - standard_error(best, n)]
