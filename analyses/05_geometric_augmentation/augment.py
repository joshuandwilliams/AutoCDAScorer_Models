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

import numpy as np
import cv2 as cv

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
    T = np.array([[1, 0, (x1 + x2 - 1) / 2 + shift_x],
                  [0, 1, (y1 + y2 - 1) / 2 + shift_y], [0, 0, 1]], dtype=np.float64)
    return (T @ H @ R @ Z @ S)[:2], (int(round(w)), int(round(h)))


def out_of_frame(raw_shape, box, **kw):
    """Fraction of the sampled window's corners that fall outside the raw image."""
    kw = {k: v for k, v in kw.items() if k not in ("flip_x", "flip_y", "brightness")}
    M, (w, h) = _matrix(box, **kw)
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]],
                       dtype=np.float64).T
    pts = M @ corners
    raw_h, raw_w = raw_shape[:2]
    outside = ((pts[0] < 0) | (pts[0] > raw_w - 1) | (pts[1] < 0) | (pts[1] > raw_h - 1))
    return float(outside.mean())


def crop(raw, box, flip_x=False, flip_y=False, brightness=1.0, **kw):
    """One augmented 64x64 crop, sampled directly out of the raw image.

    Border pixels use cv.BORDER_REFLECT_101 rather than black. Any window needing it is
    already flagged by `out_of_frame`; reflecting keeps leaf-like texture there instead of
    a black wedge, which the Otsu leaf mask would otherwise exclude -- that would change
    the measured leaf area rather than the thing the augmentation was meant to change.
    """
    M, (w, h) = _matrix(box, **kw)
    img = cv.warpAffine(raw, M, (w, h), flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP,
                        borderMode=cv.BORDER_REFLECT_101)
    img = cv.resize(img, (OUT, OUT))
    if flip_x:
        img = cv.flip(img, 0)
    if flip_y:
        img = cv.flip(img, 1)
    if brightness != 1.0:
        img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(raw.dtype)
    return img


# The augmentations to test, each applied ALONE. Magnitudes are the ones the original grid
# used (from_HPC/models/Model_0029_Aug_Sensitivity/random_search_array_sample.py,
# aug_param_grid), plus small-angle rotation variants the original grid did not contain.
AUGMENTATIONS = {
    "identity":       {},
    "flip_x":         {"flip_x": True},
    "flip_y":         {"flip_y": True},
    "rot90":          {"rotation": 90},
    "rot180":         {"rotation": 180},
    # The policy in src/augment.py draws one of FOUR right-angle rotations, so 270 is
    # applied as often as 90 and 180 and belongs in the measured set alongside them.
    "rot270":         {"rotation": 270},
    "rot15":          {"rotation": 15},
    # The policy draws every perturbation symmetrically about zero (rotation +/- 15, shift
    # +/- 10% of the box side), and shifts BOTH axes on the same draw rather than one.
    # The negative halves and the joint shift are measured here so the sweep covers
    # everything the policy can produce and does not need re-running to answer that.
    "rot-15":         {"rotation": -15},
    "shift_x-10pct":  {"shift_x_frac": -0.10},
    "shift_y-10pct":  {"shift_y_frac": -0.10},
    "shift_xy+10pct": {"shift_x_frac": 0.10, "shift_y_frac": 0.10},
    "shift_xy-10pct": {"shift_x_frac": -0.10, "shift_y_frac": -0.10},
    "rot150":         {"rotation": 150},
    "rot300":         {"rotation": 300},
    "shift_x+20":     {"shift_x": 20},
    "shift_y+20":     {"shift_y": 20},
    # The magnitude the policy in src/augment.py actually applies: a tenth of the box side
    # rather than a fixed 20 px. Resolved per crop in compute.py, because the box differs
    # from crop to crop; the value here is the fraction, not a pixel count.
    "shift_x+10pct":  {"shift_x_frac": 0.10},
    "shift_y+10pct":  {"shift_y_frac": 0.10},
    "zoom0.8":        {"zoom": 0.8},
    "zoom1.2":        {"zoom": 1.2},
    "shear_x0.3":     {"shear_x": 0.3},
    "shear_y0.3":     {"shear_y": 0.3},
    "brightness1.2":  {"brightness": 1.2},
    "brightness0.8":  {"brightness": 0.8},
}
