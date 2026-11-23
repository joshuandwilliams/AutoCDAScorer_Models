import cv2 as cv
import numpy as np
import pytest

from augment import (
    MAX_ANGLE,
    MAX_SHIFT_FRAC,
    OUT,
    PERTURBATIONS,
    ZOOM_RANGE,
    crop,
    out_of_frame,
    random_params,
)


@pytest.fixture
def raw() -> np.ndarray:
    """A 200x300 image with a different value at every pixel, so any shift shows."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)


BOX = (100, 180, 60, 140)  # x1, x2, y1, y2


class TestCrop:
    def test_identity_matches_plain_crop_and_resize(self, raw):
        x1, x2, y1, y2 = BOX
        expected = cv.resize(raw[y1:y2, x1:x2], (OUT, OUT))
        np.testing.assert_array_equal(crop(raw, BOX), expected)

    def test_output_shape_and_dtype(self, raw):
        img = crop(raw, BOX, rotation=10.0, zoom=1.1, shift_x=5, shift_y=-5)
        assert img.shape == (OUT, OUT, 3)
        assert img.dtype == np.uint8

    def test_flips(self, raw):
        base = crop(raw, BOX)
        np.testing.assert_array_equal(crop(raw, BOX, flip_x=True), cv.flip(base, 0))
        np.testing.assert_array_equal(crop(raw, BOX, flip_y=True), cv.flip(base, 1))

    def test_brightness_scales_and_clips(self, raw):
        base = crop(raw, BOX).astype(np.float32)
        bright = crop(raw, BOX, brightness=2.0)
        np.testing.assert_array_equal(bright, np.clip(base * 2.0, 0, 255).astype(np.uint8))

    def test_rotation_by_180_reverses_the_crop(self, raw):
        # Rotating about the box centre by 180 degrees is a flip in both axes
        x1, x2, y1, y2 = BOX
        native = raw[y1:y2, x1:x2]
        expected = cv.resize(cv.flip(native, -1), (OUT, OUT))
        np.testing.assert_array_equal(crop(raw, BOX, rotation=180.0), expected)


class TestOutOfFrame:
    def test_inside_frame(self, raw):
        assert out_of_frame(raw.shape, BOX) == 0.0

    def test_window_off_the_left_edge(self, raw):
        # Shifting the box 150 px left puts both left corners outside the image
        assert out_of_frame(raw.shape, BOX, shift_x=-150) == 0.5

    def test_ignores_flip_and_brightness(self, raw):
        assert out_of_frame(raw.shape, BOX, flip_x=True, flip_y=True, brightness=2.0) == 0.0


class TestRandomParams:
    def test_draws_stay_within_the_policy(self):
        rng = np.random.default_rng(1)
        side = ((BOX[1] - BOX[0]) + (BOX[3] - BOX[2])) / 2
        for _ in range(200):
            p, record = random_params(rng, BOX)
            quarter_angle = 90.0 * record["quarter"]
            assert abs(p["rotation"] - quarter_angle) <= MAX_ANGLE
            assert abs(p["shift_x"]) <= MAX_SHIFT_FRAC * side
            assert abs(p["shift_y"]) <= MAX_SHIFT_FRAC * side
            assert ZOOM_RANGE[0] <= p["zoom"] <= ZOOM_RANGE[1]
            assert p["flip_x"] == record["flip_x"]
            assert p["flip_y"] is False
            chosen = record["perturbations"].split(",")
            assert 1 <= len(chosen) <= 2
            assert set(chosen) <= set(PERTURBATIONS)

    @pytest.mark.parametrize("kind", PERTURBATIONS)
    def test_only_chosen_perturbations_change(self, kind):
        # With one perturbation, the other two stay at their identity values
        rng = np.random.default_rng(2)
        for _ in range(100):
            p, record = random_params(rng, BOX, n_perturb=1)
            if record["perturbations"] != kind:
                continue
            if kind != "shift":
                assert p["shift_x"] == 0 and p["shift_y"] == 0
            if kind != "zoom":
                assert p["zoom"] == 1.0
            if kind != "rot_small":
                assert p["rotation"] == 90.0 * record["quarter"]

    def test_three_perturbations(self):
        p, record = random_params(np.random.default_rng(3), BOX, n_perturb=3)
        assert record["perturbations"] == ",".join(sorted(PERTURBATIONS))

    def test_output_feeds_crop(self, raw):
        p, _ = random_params(np.random.default_rng(4), BOX)
        assert crop(raw, BOX, **p).shape == (OUT, OUT, 3)
