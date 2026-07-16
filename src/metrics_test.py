
from metrics import near_miss_accuracy, quadratic_weighted_kappa


class TestNearMissAccuracy:
    def test_exact_and_near(self):
        # diffs: 0, 1, 2, 0 -> within tolerance 1: True, True, False, True -> 0.75
        y_true = [0, 1, 2, 3]
        y_pred = [0, 2, 4, 3]
        assert near_miss_accuracy(y_true, y_pred) == 0.75

    def test_perfect(self):
        assert near_miss_accuracy([0, 1, 2], [0, 1, 2]) == 1.0

    def test_tolerance_zero_is_exact_accuracy(self):
        assert near_miss_accuracy([0, 1, 2, 3], [0, 2, 2, 3], tolerance=0) == 0.75

    def test_empty(self):
        assert near_miss_accuracy([], []) == 0.0


class TestQuadraticWeightedKappa:
    def test_perfect_agreement(self):
        labels = list(range(7))
        assert quadratic_weighted_kappa(labels, labels, labels=labels) == 1.0

    def test_returns_float_and_penalises_far_errors_more(self):
        labels = list(range(7))
        near = quadratic_weighted_kappa([0, 1, 2, 3], [1, 2, 3, 4], labels=labels)
        far = quadratic_weighted_kappa([0, 1, 2, 3], [6, 5, 4, 6], labels=labels)
        assert isinstance(near, float)
        assert near > far
