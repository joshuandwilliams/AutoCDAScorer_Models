"""Candidate figures for the cross-family ensembling result (thesis #fig-cross-model-agreement).

Every variant is built from the same numbers as ensembling_results.qmd section 4: the
tied set per family, softmax averaged within a family then across families, ensemble
split used to choose and test scored once. Run from analyses/14_ensembling.
"""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path.cwd().parents[1]
OUT = Path("figure_options")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

Z = np.load("pred_cache/probs.npz", allow_pickle=True)
ye, yt = Z["ye"].astype(int), Z["yt"].astype(int)
fe = [str(f).replace(".tif", "") for f in Z["files_e"]]
ft = [str(f).replace(".tif", "") for f in Z["files_t"]]


def SE(p, n=2289):
    return float(np.sqrt(p * (1 - p) / n))


GRIDS = {
    "CNN": ("models/03_Medoid_CNN_RandomSearch/01_Categorical_Loss", "cnn", "avg_vaf"),
    "ViT": ("models/04_Medoid_ViT_RandomSearch/01_Categorical_Loss", "vit", "avg_vaf"),
    "CNN\n(geom. aug)": (
        "models/07_Medoid_CNN_GeomAug/01_MatchLargestClass",
        "geo",
        "avg_vaf_orig",
    ),
}
FAM = {}
for name, (bd, tag, col) in GRIDS.items():
    g = ROOT / bd
    r = (
        pd.concat(
            [pd.read_csv(g / "random_search_results.csv")]
            + [pd.read_csv(q) for q in (g / "parts").glob("*.csv")],
            ignore_index=True,
        )
        .drop_duplicates("model_id")
        .set_index("model_id")
    )
    w = r[col].dropna()
    tie = set(w[w >= w.max() - SE(w.max())].index)
    k = [i for i, m in enumerate(Z[f"{tag}_ids"]) if m in tie]
    FAM[name] = (Z[f"{tag}_e"][k], Z[f"{tag}_t"][k])
PC = [f"p{i}" for i in range(7)]
nd = pd.read_csv(ROOT / "analyses/03_feature_classifiers/model_cache/nondl_heldout_predictions.csv")
FAM["Ordinal\nregression"] = (
    nd[nd.split == "ensemble"].set_index("image_id").reindex(fe)[PC].to_numpy(float)[None],
    nd[nd.split == "test"].set_index("image_id").reindex(ft)[PC].to_numpy(float)[None],
)

N = list(FAM)


def ex(p, y):
    return float(np.mean(p == y))


H = {k: v[0].argmax(2) for k, v in FAM.items()}


def comb(keys, w):
    return np.stack([FAM[k][w].mean(0) for k in keys]).mean(0).argmax(1)


# Matrices. Diagonal = that family alone; off-diagonal = the two combined.
n = len(N)
T = np.zeros((n, n))
E = np.zeros((n, n))
Q = np.full((n, n), np.nan)
D = np.full((n, n), np.nan)
for i, a in enumerate(N):
    T[i, i] = ex(FAM[a][1].mean(0).argmax(1), yt)
    E[i, i] = ex(FAM[a][0].mean(0).argmax(1), ye)
for i, j in itertools.combinations(range(n), 2):
    a, b = N[i], N[j]
    T[i, j] = T[j, i] = ex(comb((a, b), 1), yt)
    E[i, j] = E[j, i] = ex(comb((a, b), 0), ye)
    qs, ds = [], []
    for p in H[a]:
        for q_ in H[b]:
            oa, ob = p == ye, q_ == ye
            aa = (oa & ob).mean()
            bb = (~oa & ob).mean()
            cc = (oa & ~ob).mean()
            dd = (~oa & ~ob).mean()
            qs.append((aa * dd - bb * cc) / (aa * dd + bb * cc))
            ds.append(dd)
    Q[i, j] = Q[j, i] = np.mean(qs)
    D[i, j] = D[j, i] = np.mean(ds)
for i in range(n):  # within-family Q on the diagonal
    if len(H[N[i]]) > 1:
        qs = []
        for x, y_ in itertools.combinations(range(len(H[N[i]])), 2):
            oa, ob = H[N[i]][x] == ye, H[N[i]][y_] == ye
            aa = (oa & ob).mean()
            bb = (~oa & ob).mean()
            cc = (oa & ~ob).mean()
            dd = (~oa & ~ob).mean()
            qs.append((aa * dd - bb * cc) / (aa * dd + bb * cc))
        Q[i, i] = np.mean(qs)

HUMAN = 0.6892
BEST_SINGLE = 0.6461  # best single CNN chosen on the ensemble split, section 3


def heat(ax, M, title, fmt="{:.1f}", pct=True, cmap="RdYlBu_r", centre=None, cbar_label=""):
    V = 100 * M if pct else M
    if centre is not None:
        norm = TwoSlopeNorm(vcenter=centre, vmin=np.nanmin(V), vmax=np.nanmax(V))
        im = ax.imshow(V, cmap=cmap, norm=norm)
    else:
        im = ax.imshow(V, cmap=cmap)
    ax.set_xticks(range(n), N, fontsize=8)
    ax.set_yticks(range(n), N, fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isnan(V[i, j]):
                continue
            r, g, b, _ = im.cmap(im.norm(V[i, j]))
            ax.text(
                j,
                i,
                fmt.format(V[i, j]),
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "black",
                fontweight="bold" if i != j else "normal",
            )
    ax.set_title(title, fontsize=9.5)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(cbar_label, fontsize=8)
    return im


# ---- Variant A: one heatmap, test accuracy only, diverging about the best single model
fig, ax = plt.subplots(figsize=(5.0, 4.2))
heat(
    ax,
    T,
    "Test exact accuracy (%)\ndiagonal = family alone, off-diagonal = the two combined",
    centre=100 * BEST_SINGLE,
    cbar_label="% (white = best single model, 64.6)",
)
fig.tight_layout()
fig.savefig(OUT / "A_test_only.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- Variant B: two heatmaps, ensemble split and test
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
heat(axes[0], E, "A  Ensemble split (used to choose)", centre=100 * BEST_SINGLE, cbar_label="%")
heat(axes[1], T, "B  Test split (scored once)", centre=100 * BEST_SINGLE, cbar_label="%")
fig.tight_layout()
fig.savefig(OUT / "B_two_heatmaps.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ---- Variant C: lower triangle only, both numbers per cell, plain linear colour scale.
# TwoSlopeNorm is deliberately NOT used here: it maps the 1.3 pp below the best single
# model onto the same colour span as the 4.6 pp above it, which pushed 63.3 to the far end
# of the bar. A plain Normalize keeps the gaps proportional to the differences.
def truncate(name, lo=0.18, hi=0.95):
    """Trim the pale end off a colour map so the lowest cell is still visibly filled
    rather than blending into the white page."""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(f"{name}_t", base(np.linspace(lo, hi, 256)))


def tri_panel(cmap, fname, lo=0.18, hi=0.95, figsize=(5.2, 4.4)):
    M = np.where(np.tril(np.ones((n, n)), 0) == 1, T, np.nan)
    V = 100 * M
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(V, cmap=truncate(cmap, lo, hi), vmin=np.nanmin(V), vmax=np.nanmax(V))
    ax.set_xticks(range(n), N, fontsize=8)
    ax.set_yticks(range(n), N, fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            r, g, b, _ = im.cmap(im.norm(V[i, j]))
            c = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"
            ax.text(
                j, i - 0.08, f"{100 * T[i, j]:.1f}", ha="center", va="center", fontsize=11, color=c
            )
            ax.text(
                j,
                i + 0.26,
                f"({100 * E[i, j]:.1f})",
                ha="center",
                va="center",
                fontsize=7.5,
                color=c,
            )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Test exact accuracy (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Diverging blue-to-red. Both halves are used across the full linear range, so the
# midpoint of the colour bar is the midpoint of the data, not the best single model.
tri_panel("RdBu_r", "C1_bluered.png", lo=0.10, hi=0.90)
tri_panel("coolwarm", "C2_coolwarm.png", lo=0.05, hi=0.95)
tri_panel("Blues", "C3_blues.png")
tri_panel("Greys", "C4_greys.png")

# ---- Variant D: heatmap of test accuracy + scatter of error correlation vs gain
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
heat(axes[0], T, "A  Test exact accuracy (%)", centre=100 * BEST_SINGLE, cbar_label="%")
ax = axes[1]
pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)]
xs = [Q[i, j] for i, j in pairs]
ys = [100 * T[i, j] for i, j in pairs]
wx = [Q[i, i] for i in range(n) if not np.isnan(Q[i, i])]
wy = [100 * T[i, i] for i in range(n) if not np.isnan(Q[i, i])]
ax.scatter(wx, wy, s=55, facecolors="none", edgecolors="#C44E52", label="One family alone")
ax.scatter(xs, ys, s=55, color="#4C72B0", label="Two families combined")
for (i, j), x, y in zip(pairs, xs, ys, strict=True):
    ax.annotate(
        f"{N[i]} + {N[j]}".replace("\n", " "),
        (x, y),
        fontsize=6.5,
        xytext=(4, 4),
        textcoords="offset points",
    )
ax.axhline(100 * HUMAN, ls="--", lw=1.1, color="#C44E52")
ax.text(
    ax.get_xlim()[0], 100 * HUMAN + 0.12, " human raters", fontsize=7, color="#C44E52", va="bottom"
)
ax.set_xlabel("Yule's Q between the members (lower = errors less related)")
ax.set_ylabel("Test exact accuracy (%)")
ax.set_title("B  Less-related errors, higher accuracy", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "D_heatmap_plus_scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- Variant E: two heatmaps, Yule's Q and test accuracy
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
heat(
    axes[0],
    Q,
    "A  Yule's Q between members\n(lower = errors less related)",
    fmt="{:.3f}",
    pct=False,
    cmap="RdYlBu_r",
    cbar_label="Q",
)
heat(axes[1], T, "B  Test exact accuracy (%)", centre=100 * BEST_SINGLE, cbar_label="%")
fig.tight_layout()
fig.savefig(OUT / "E_Q_and_test.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(
    pd.DataFrame(
        100 * T, index=[x.replace("\n", " ") for x in N], columns=[x.replace("\n", " ") for x in N]
    )
    .round(2)
    .to_string()
)
print(
    "\nYule's Q\n",
    pd.DataFrame(
        Q, index=[x.replace("\n", " ") for x in N], columns=[x.replace("\n", " ") for x in N]
    )
    .round(3)
    .to_string(),
)
print("\nwrote:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
