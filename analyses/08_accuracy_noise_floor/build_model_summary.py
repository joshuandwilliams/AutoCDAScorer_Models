"""Build the chapter's model-summary figure (thesis #fig-model-summary).

Every deep row is the setting with the best cross-validated accuracy in its grid, with an
error bar spanning the settings tied within one standard error of it. Each grid is first
collapsed to its distinct settings, since the CNN grids train reg=None once per
reg_strength and a raw tie band would count one setting up to three times. The two ensemble
ensemble row comes from analyses/14_ensembling: the best cross-family pair, selected on the
455-crop ensemble split and scored once on test.

    conda run -n autocdascorer-analyses python build_model_summary.py
"""

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import offset_copy

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from grid_results import deduplicate  # noqa: E402

OUT = Path(__file__).resolve().parent / "thesis_plots"
OUT.mkdir(parents=True, exist_ok=True)

HUMAN_EXACT, HUMAN_NEAR = 0.6892, 0.9280
HUM_LW = 1.4  # width of the human reference line, in points


def grid(rel, rank="avg_vaf"):
    """Best configuration by `rank`, with the spread over the configurations tied with it.

    models/07 is ranked on avg_vaf like every unaugmented grid: its pool is topped up to
    1,627 crops per class, so avg_vaf carries the same class balance as the 308-crop test
    split, while avg_vaf_orig is measured on the originals alone and those are 42.6% score
    0. models/08 keeps avg_vaf_orig, because only 10.9% of the cGAN pool is real and its
    avg_vaf is therefore largely accuracy on generated crops.
    """
    base = ROOT / rel
    files = [base / "random_search_results.csv"] + [
        Path(f) for f in glob.glob(str(base / "parts/*.csv"))
    ]
    d = deduplicate(
        pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
        .drop_duplicates("model_id")
        .dropna(subset=[rank]),
        rank,
    )
    best = d.loc[d[rank].idxmax()]
    se = float(np.sqrt(best[rank] * (1 - best[rank]) / 2289))
    tied = d[d[rank] >= best[rank] - se]
    return dict(
        exact=best.test_acc,
        near=best.test_nearmiss,
        elo=tied.test_acc.min(),
        ehi=tied.test_acc.max(),
        nlo=tied.test_nearmiss.min(),
        nhi=tied.test_nearmiss.max(),
        n=len(tied),
    )


ROWS = [
    ("Human raters", dict(exact=HUMAN_EXACT, near=HUMAN_NEAR)),
    ("Ordinal regression", dict(exact=0.6331, near=0.9643)),
    ("CNN", grid("models/03_Medoid_CNN_RandomSearch/01_Categorical_Loss")),
    ("CNN, soft labels", grid("models/05_Medoid_CNN_SoftLabels/01_Categorical_Loss")),
    ("ViT", grid("models/04_Medoid_ViT_RandomSearch/01_Categorical_Loss")),
    (
        "CNN, geometric aug",
        grid("models/07_Medoid_CNN_GeomAug/01_MatchLargestClass"),
    ),
    ("CNN, cGAN aug", grid("models/08_Medoid_CNN_CGANAug/01_MatchLargestClass", "avg_vaf_orig")),
    # The chapter's selected best single model. The geometric-aug row above is that grid's
    # top configuration on avg_vaf (7927); the chapter breaks that tie on the held-out
    # ensemble split and selects 6848 instead, so it needs its own row. No error bar: it is
    # one fixed model rather than a configuration drawn from a grid.
    ("Best single model:\nCNN, geometric aug", dict(exact=0.6688, near=0.9903)),
    # analyses/14_ensembling, sections 3 and 4. No error bar: each is one fixed rule over a
    # fixed member set, not a configuration drawn from a grid.
    ("Ensemble: geometric aug\n+ ordinal regression", dict(exact=0.6883, near=0.9838)),
]

# Sorted by exact accuracy, with the human reference forced to the top row so it reads as
# the benchmark rather than as another entry in the ranking.
order = sorted([r for r in ROWS if r[0] != "Human raters"], key=lambda kv: kv[1]["exact"])
order += [r for r in ROWS if r[0] == "Human raters"]
names = [k for k, _ in order]

FAMILY = {
    "Human raters": "human",
    "Ordinal regression": "feature",
    "Ensemble: geometric aug\n+ ordinal regression": "ensemble",
}
COL = {
    "human": "#F3B5B7",
    "feature": "#F6CBA6",
    "deep": "#AFC8E6",
    "ensemble": "#B9DEB4",
}
fam = [FAMILY.get(n, "deep") for n in names]

fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.0))
for ax, (key, lo, hi, lab, hum) in zip(
    axes,
    [
        ("exact", "elo", "ehi", "Exact accuracy", HUMAN_EXACT),
        ("near", "nlo", "nhi", "Near-miss accuracy", HUMAN_NEAR),
    ],
    strict=True,
):
    y = np.arange(len(order))
    v = np.array([d[key] for _, d in order])
    ax.barh(y, 100 * v, color=[COL[f] for f in fam], height=0.68, zorder=2)
    # Error bars only on the rows that have a tied set. Passing a zero-width xerr to barh
    # for the others still draws a cap, which reads as a range of zero rather than as no
    # range at all.
    idx = [i for i, (_, d) in enumerate(order) if lo in d]
    if idx:
        ax.errorbar(
            100 * v[idx],
            np.array(y)[idx],
            fmt="none",
            xerr=100
            * np.array(
                [
                    [order[i][1][key] - order[i][1][lo] for i in idx],
                    [order[i][1][hi] - order[i][1][key] for i in idx],
                ]
            ),
            ecolor="#1A1A1A",
            elinewidth=1.6,
            capsize=4,
            capthick=1.6,
        )
    # The line is offset left by half its own width so its RIGHT edge lands on the human
    # value rather than straddling it. axvline centres the stroke on the coordinate, which
    # pushes half the linewidth past the human bar's tip and makes anything just above the
    # human rate look level with it. get_xaxis_transform is x-in-data, y-in-axes.
    tr = offset_copy(ax.get_xaxis_transform(), fig=fig, x=-HUM_LW / 2, y=0, units="points")
    ax.add_line(
        Line2D([100 * hum] * 2, [0, 1], transform=tr, color="#C44E52", ls="--", lw=HUM_LW, zorder=3)
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(f"{lab} on the 308 test CDAs (%)")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.25, axis="x")
    # Keep the right spine: it marks 100%, so how far each bar falls short is readable
    # without counting gridlines.
    ax.margins(y=0.02)

handles = [
    Patch(color=COL["deep"], label="Deep learning"),
    Patch(color=COL["ensemble"], label="Ensemble"),
    Patch(color=COL["feature"], label="Chapter 2 features"),
    Patch(color=COL["human"], label="Human raters"),
    Line2D([], [], color="#1A1A1A", lw=1.6, label="Range over configurations tied within 1 SE"),
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8.5,
    bbox_to_anchor=(0.5, -0.005),
)
fig.tight_layout(rect=(0, 0.07, 1, 1))

# Each axis is written on its own so the thesis assembler can letter them A and B.
fig.canvas.draw()
r = fig.canvas.get_renderer()
for nm, a in zip("ab", axes, strict=True):
    bb = a.get_tightbbox(r).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(OUT / f"model_summary_{nm}.png", dpi=450, bbox_inches=bb)
lf = plt.figure(figsize=(7.2, 0.5))
lf.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=8.5)
lf.savefig(OUT / "model_summary_legend.png", dpi=450, bbox_inches="tight")

for n, d in order:
    print(
        f"{n.replace(chr(10), ' '):42s} exact {100 * d['exact']:6.2f}  near {100 * d['near']:6.2f}"
        + (f"   tied {d['n']:3d}" if "n" in d else "")
    )
