"""The Chapter 2 hand-crafted crop features, ported to Python.

Why this exists
---------------
Every other number in `gan_diversity.qmd` is measured inside the feature space of a CNN
trained on real crops, so generated images are judged through a lens fitted to their
competitor. That could bias the comparison in either direction and there is no way to
argue it away from inside that space.

These twelve features are fixed arithmetic. Nothing about them is fitted to real crops or
to generated ones -- they are Otsu segmentation plus means, spreads and counts over the
resulting regions -- so applying them to both sets gives a comparison with no learned
component anywhere in it. If the CNN-feature result and this one agree, the lens is not
driving the conclusion.

Provenance
----------
Ported line-for-line from
`uv-agroinfiltration-dataset/analyses/10_segmentation_and_feature_extraction/`
(section 1.3 for the segmentation, section 5.1 for the twelve features). The R original
uses EBImage; the equivalences used here are:

    EBImage::gblur(x, sigma)        -> ndimage.gaussian_filter(x, sigma, truncate=3.0)
    EBImage::otsu(v, range=(0,1))   -> _otsu_256 below (256 bins over [0, 1])
    EBImage::medianFilter(m, r)     -> ndimage.median_filter(m, size=2*r+1)
    EBImage::fillHull(m)            -> ndimage.binary_fill_holes(m)
    EBImage::bwlabel + largest      -> ndimage.label + argmax of the component sizes

Channel order
-------------
The R original reads TIFFs with readTIFF, which returns RGB. Here both the real crops
(`data/datasets/medoid_64/medoid_64.npy`, built by `base_dataset` via `cv.imread`) and the
generated crops (read with `cv.imread`) are **BGR**, so `greenness` is channel 1 minus
channel 2 rather than 1 minus 0. `gan_diversity.qmd` checks this empirically rather than
taking it on trust -- mean greenness must rise with score on the real crops, whose labels
are known correct.
"""

import numpy as np
from scipy import ndimage

FEATURE_NAMES = [
    "sep", "green_gap", "cda_green", "frac_g_gt_r", "cda_green_sd", "cda_bright",
    "cda_bright_sd", "leaf_green_sd", "mean_green", "log_gr", "leaf_bright", "ent_gap",
]

# What each feature measures, and which way it should move with severity. Taken from the
# Chapter 2 table so the direction checks in the qmd have something to test against.
FEATURE_DOC = {
    "sep":           ("leaf-greenness bimodality (Otsu between-class / total variance)", "rises"),
    "green_gap":     ("CDA greenness minus healthy-leaf greenness", "rises"),
    "cda_green":     ("mean greenness inside the CDA", "rises"),
    "frac_g_gt_r":   ("lesion size: |CDA| / |leaf|", "rises"),
    "cda_green_sd":  ("greenness spread within the CDA", "rises"),
    "cda_bright":    ("mean brightness inside the CDA", "rises"),
    "cda_bright_sd": ("brightness spread within the CDA", "rises"),
    "leaf_green_sd": ("greenness spread over the whole leaf", "rises"),
    "mean_green":    ("mean green-channel intensity over the leaf", "rises"),
    "log_gr":        ("log ratio of green-dominant to red-dominant leaf pixels", "rises"),
    "leaf_bright":   ("mean brightness over the leaf", "rises"),
    "ent_gap":       ("CDA minus healthy-leaf Shannon entropy", "rises"),
}


def leaf_mask(img):
    """Leaf region: non-black pixels. Drops the near-black background of a crop."""
    return img.mean(axis=2) >= 0.15


def greenness(img):
    """Green minus red. Healthy leaf is strongly negative, yellow-green CDA is near zero.

    Channel 2 is red because the arrays are BGR -- see the module docstring.
    """
    return img[:, :, 1] - img[:, :, 2]


def brightness(img):
    return img.mean(axis=2)


def _otsu_256(v):
    """EBImage::otsu(v, range=(0, 1)): Otsu's threshold on a 256-bin histogram of [0, 1]."""
    h, edges = np.histogram(np.clip(v, 0, 1), bins=256, range=(0.0, 1.0))
    p = h / max(h.sum(), 1)
    mids = (edges[:-1] + edges[1:]) / 2
    w1 = np.cumsum(p)
    m1 = np.cumsum(p * mids)
    mu = m1[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        bcv = (mu * w1 - m1) ** 2 / (w1 * (1 - w1) + 1e-12)
    return float(edges[int(np.nanargmax(bcv)) + 1])


def otsu_sep(x, nbins=64):
    """Otsu threshold plus 'separability' = between-class / total variance, in [0, 1].

    High separability means the values split cleanly into two modes, i.e. the leaf really
    does contain two populations (lesion and healthy tissue) rather than one.
    """
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-6:
        return {"thr": lo, "sep": 0.0, "frac_hi": 0.0}
    edges = np.linspace(lo, hi, nbins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    h, _ = np.histogram(x, bins=edges)
    p = h / h.sum()
    mu = float((p * mids).sum())
    tot = float((p * (mids - mu) ** 2).sum())
    w1 = np.cumsum(p)
    m1 = np.cumsum(p * mids)
    with np.errstate(divide="ignore", invalid="ignore"):
        bcv = (mu * w1 - m1) ** 2 / (w1 * (1 - w1) + 1e-12)
    k = int(np.nanargmax(bcv))
    return {"thr": float(edges[k + 1]),
            "sep": float(np.nanmax(bcv) / (tot + 1e-12)),
            "frac_hi": float(np.mean(x > edges[k + 1]))}


def segment_cda(img, sigma=2):
    """Otsu on blurred within-leaf greenness, then morphological cleanup.

    Returns the leaf mask and the CDA mask. The cleanup radius scales with crop size
    exactly as the R original does (5 px at the 264-px median crop), which at 64x64 gives
    a 5x5 median filter.
    """
    leaf = leaf_mask(img)
    g = greenness(img)
    gs = (g - g.min()) / (g.max() - g.min() + 1e-9)
    gb = ndimage.gaussian_filter(gs, sigma=sigma, truncate=3.0)
    if not leaf.any():
        return leaf, np.zeros_like(leaf)
    thr = _otsu_256(gb[leaf])
    mask = leaf & (gb > thr)

    b = max(1, int(round(5 * np.mean(img.shape[:2]) / 264)))
    mask = ndimage.binary_fill_holes(ndimage.median_filter(mask, size=2 * (2 * b) + 1))
    # The median filter and hole-fill ignore the leaf boundary, so re-apply the leaf mask:
    # the CDA can never extend onto the background.
    mask = mask & leaf
    lab, n = ndimage.label(mask)
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]
        mask = lab == (int(np.argmax(sizes)) + 1)
    # Re-applying the leaf mask punches holes back in wherever the leaf is dark inside the
    # lesion. Fill those so the CDA is one solid region.
    return leaf, ndimage.binary_fill_holes(mask)


def _shannon_entropy(v):
    q = np.round(np.clip(v, 0, 1) * 255).astype(int)
    counts = np.bincount(q, minlength=256)
    p = counts[counts > 0] / len(q)
    return float(-(p * np.log2(p)).sum())


def region_entropy(img, mask):
    if not mask.any():
        return np.nan
    return _shannon_entropy(np.concatenate([img[:, :, c][mask] for c in range(3)]))


def features(img):
    """The twelve Chapter 2 features for one crop, as a dict.

    All region features are read off the SAME segmentation -- leaf, CDA, and healthy leaf
    (leaf minus CDA). Nothing is re-thresholded per feature.
    """
    leaf, cda = segment_cda(img)
    cda = cda & leaf
    healthy = leaf & ~cda
    gmat, bmat = greenness(img), brightness(img)
    G, R = img[:, :, 1], img[:, :, 2]

    def mn(x):
        return float(np.mean(x)) if x.size else 0.0

    def nsd(x):
        return float(np.std(x, ddof=1)) if x.size > 1 else 0.0

    gv_leaf = gmat[leaf]
    o = otsu_sep(gv_leaf) if gv_leaf.size else {"sep": 0.0}
    n_green = int(np.sum(G[leaf] >= R[leaf]))
    n_red = int(np.sum(R[leaf] > G[leaf]))
    both = cda.any() and healthy.any()

    return {
        "sep":           o["sep"],
        "green_gap":     mn(gmat[cda]) - mn(gmat[healthy]) if both else 0.0,
        "cda_green":     mn(gmat[cda]),
        "frac_g_gt_r":   cda.sum() / max(leaf.sum(), 1),
        "cda_green_sd":  nsd(gmat[cda]) * 255,
        "cda_bright":    mn(bmat[cda]) * 255,
        "cda_bright_sd": nsd(bmat[cda]) * 255,
        "leaf_green_sd": nsd(gv_leaf) * 255,
        "mean_green":    mn(G[leaf]) * 255,
        "log_gr":        float(np.log((n_green + 1) / (n_red + 1))),
        "leaf_bright":   mn(bmat[leaf]) * 255,
        "ent_gap":       (region_entropy(img, cda) - region_entropy(img, healthy)) if both else 0.0,
    }


def feature_table(images):
    """Stack `features` over an iterable of crops into an (n, 12) array."""
    return np.array([[features(im)[k] for k in FEATURE_NAMES] for im in images],
                    dtype="float64")
