"""Single geometric augmentations applied to a CDA bounding box, sampled from the raw image.

Why sample from the raw image
-----------------------------
Rotating or shearing an already-cropped image leaves black corners, and a shift or zoom
applied to a crop can only pad, never reveal. The original pipeline avoided this by
re-cropping a larger region from the raw leaf photograph before transforming
(`from_HPC/models/Model_0029_Aug_Sensitivity/random_search_array.py`,
`combine_rotate_shear` STEP 2-4). The same intent is kept here, implemented as one inverse
affine map from the 64x64 output grid back into raw-image coordinates so every output
pixel is real leaf, with no corner filling and no repeated resampling.

Difference from the original, deliberate
----------------------------------------
The original clamped out-of-bounds windows with `check_coords`, which does not work. Each
of its four branches updates one coordinate and then computes the second from the
*already-updated* first, so the correction is always zero:

    if x1 < 0:
        x1 = x1 + abs(x1)      # x1 is now 0
        x2 = x2 + abs(x1)      # abs(x1) is now 0, so x2 never moves

The branches are also chained with `elif`, so a window hanging off two edges gets only one
of them treated. The net effect is that a window overhanging the frame is silently
narrowed on one side instead of being translated back inside, which changes the field of
view and the aspect ratio before the resize to 64x64 stretches it.

Here the window is instead translated back inside the frame, and `out_of_frame` reports
how much of the requested window fell outside so the two can be told apart.
"""

import cv2 as cv
import numpy as np

OUT = 64


def _matrix(box, shift_x=0, shift_y=0, zoom=1.0, rotation=0.0, shear_x=0.0, shear_y=0.0):
    """Affine map from native-resolution output pixels to raw-image pixels.

    Composed in box-centred units so every transform turns about the lesion's own centre
    rather than the raw frame's, which is what the original did by recentring on
    ``(x1 + x2) / 2, (y1 + y2) / 2`` at each step.

    The output grid is the box's own pixel size, not 64x64. Warping at native resolution
    and resizing afterwards makes the identity case reduce exactly to
    ``raw[y1:y2, x1:x2]`` followed by ``cv.resize(..., (64, 64))``, which is what
    ``base_dataset`` does to build the real dataset. Warping straight to 64x64 instead
    would resample a ~248 px box down by a factor of four inside ``warpAffine``, and the
    identity crop would then not match the crops the CNN was trained on.
    """
    x1, x2, y1, y2 = box
    w, h = x2 - x1, y2 - y1
    # Output pixel (u, v) -> box-centred units. Using (w - 1) / 2 rather than w / 2 puts
    # output pixel CENTRES on raw pixel CENTRES; with w / 2 the whole crop sits half a
    # pixel off, which survives the resize to 64x64 as a visible shift.
    S = np.array([[1, 0, -(w - 1) / 2], [0, 1, -(h - 1) / 2], [0, 0, 1]], dtype=np.float64)
    # zoom > 1 means "zoom in", so the sampled window shrinks by 1/zoom
    Z = np.diag([1.0 / zoom, 1.0 / zoom, 1.0])
    th = np.radians(rotation)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    H = np.array([[1, shear_x, 0], [shear_y, 1, 0], [0, 0, 1]], dtype=np.float64)
    T = np.array(
        [[1, 0, (x1 + x2 - 1) / 2 + shift_x], [0, 1, (y1 + y2 - 1) / 2 + shift_y], [0, 0, 1]],
        dtype=np.float64,
    )
    return (T @ H @ R @ Z @ S)[:2], (int(round(w)), int(round(h)))


def out_of_frame(raw_shape, box, **kw):
    """Fraction of the sampled window's corners that fall outside the raw image."""
    kw = {k: v for k, v in kw.items() if k not in ("flip_x", "flip_y", "brightness")}
    M, (w, h) = _matrix(box, **kw)
    corners = np.array(
        [[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=np.float64
    ).T
    pts = M @ corners
    raw_h, raw_w = raw_shape[:2]
    outside = (pts[0] < 0) | (pts[0] > raw_w - 1) | (pts[1] < 0) | (pts[1] > raw_h - 1)
    return float(outside.mean())


def crop(raw, box, flip_x=False, flip_y=False, brightness=1.0, **kw):
    """One augmented 64x64 crop, sampled directly out of the raw image.

    Border pixels use cv.BORDER_REFLECT_101 rather than black. Any window needing it is
    already flagged by `out_of_frame`; reflecting keeps leaf-like texture there instead of
    a black wedge, which the Otsu leaf mask would otherwise exclude -- that would change
    the measured leaf area rather than the thing the augmentation was meant to change.
    """
    M, (w, h) = _matrix(box, **kw)
    img = cv.warpAffine(
        raw,
        M,
        (w, h),
        flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
        borderMode=cv.BORDER_REFLECT_101,
    )
    img = cv.resize(img, (OUT, OUT))
    if flip_x:
        img = cv.flip(img, 0)
    if flip_y:
        img = cv.flip(img, 1)
    if brightness != 1.0:
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(raw.dtype)
    return img


# ---------------------------------------------------------------------------------------
# The augmentation policy
# ---------------------------------------------------------------------------------------
#
# Chosen from the measurements in analyses/05_geometric_augmentation. Two groups, treated
# differently because they are different kinds of thing.
#
# DIHEDRAL (free, always applied). The eight combinations of a 90-degree rotation and an
# optional flip are permutations of the pixel grid. Every one of the twelve Chapter 2
# features is a mean, a spread, a count or a threshold over a region, none of which depends
# on where in the frame a pixel sits, so all eight leave the crop identical in every
# measured property -- measured displacement 0.000 for the flips and 180 degrees, 0.008 for
# 90 degrees. They cannot mislabel anything. The top medoid CNN nevertheless changes its
# prediction on 17-22% of crops under them, so there is a real error mode to remove and it
# costs nothing to remove it. Applied to every augmented crop and not counted against the
# budget below.
#
# PERTURBATIONS (budgeted, 1-2 per crop). These move the crop somewhere it could genuinely
# have been, and their magnitudes come from how much the three human raters disagreed about
# the bounding box rather than from a guess:
#
#   shift   +/- 10% of the box side. Rater box centres differ from their 3-rater mean by a
#           median 5.0% of the box side and a 90th percentile of 10%. The fraction is flat
#           across box size (0.054, 0.050, 0.049, 0.048 by size quartile) while the pixel
#           offset is not (9.9 to 16.6 px), which is why this is a fraction and not the
#           fixed 20 px the original pipeline used -- boxes run from 82 to 524 px.
#   zoom    0.8 to 1.2. Rater box sizes span 0.814 to 1.201 of their 3-rater mean at the
#           5th and 95th percentiles, so this is the observed disagreement, not a guess.
#   rot_small  +/- 15 degrees. Best displacement-per-label-cost measured (0.136 at 0.919
#           label preservation, against 0.184 at 0.783 for the original grid's 150 degrees).
#
# Each extra perturbation costs roughly three points of label preservation, so the budget
# is 1-2 rather than the 5.69 the original pipeline was applying.
#
# DELIBERATELY EXCLUDED:
#   shear        Measured harmless (0.144 displacement at 0.902) but every leaf in this
#                dataset was photographed flat, so there is no oblique viewing for it to
#                model. It would teach invariance to a distortion that never occurs.
#   rotation at 150 / 300 degrees   Same displacement as 15 degrees at nine points worse
#                label preservation. A 150-degree rotation is an exact 180 composed with a
#                30, and the 180 is free, so the large angle buys nothing.
#   brightness   Largest displacement in the table (0.798, 0.833) and the worst label
#                preservation (0.432, 0.567). Three features are absolute intensities
#                (leaf_bright, mean_green, cda_bright) and all three correlate with
#                severity (rho 0.454, 0.682, 0.702) because dying tissue is brighter, so a
#                gain moves a crop along the severity axis. Holding the label fixed then
#                asserts something false. Exposure shift is handled by feature choice
#                instead -- see analyses/05_geometric_augmentation section 6.

PERTURBATIONS = ("shift", "zoom", "rot_small")
MAX_SHIFT_FRAC = 0.10
ZOOM_RANGE = (0.8, 1.2)
MAX_ANGLE = 15.0


def random_params(rng, box, n_perturb=None):
    """Sample one augmentation: a random dihedral element plus 1-2 perturbations.

    ``box`` is needed because the shift is a fraction of the box side rather than a fixed
    pixel count. Returns a kwargs dict for `crop`, plus a record of what was drawn.
    """
    x1, x2, y1, y2 = box
    side = ((x2 - x1) + (y2 - y1)) / 2

    p = {"rotation": 0.0, "shift_x": 0, "shift_y": 0, "zoom": 1.0, "flip_x": False, "flip_y": False}

    # Dihedral element: one of 4 right-angle rotations, optionally flipped. Free.
    quarter = int(rng.integers(4))
    p["rotation"] = 90.0 * quarter
    p["flip_x"] = bool(rng.integers(2))

    n = int(rng.integers(1, 3)) if n_perturb is None else n_perturb
    chosen = rng.choice(len(PERTURBATIONS), size=n, replace=False)
    for c in chosen:
        kind = PERTURBATIONS[c]
        if kind == "shift":
            p["shift_x"] = float(rng.uniform(-MAX_SHIFT_FRAC, MAX_SHIFT_FRAC) * side)
            p["shift_y"] = float(rng.uniform(-MAX_SHIFT_FRAC, MAX_SHIFT_FRAC) * side)
        elif kind == "zoom":
            p["zoom"] = float(rng.uniform(*ZOOM_RANGE))
        else:
            # Added to the right-angle rotation, so the total stays within one dihedral
            # element of upright and only the small part ever needs interpolation.
            p["rotation"] += float(rng.uniform(-MAX_ANGLE, MAX_ANGLE))
    return p, {
        "quarter": quarter,
        "flip_x": p["flip_x"],
        "perturbations": ",".join(sorted(PERTURBATIONS[c] for c in chosen)),
    }
