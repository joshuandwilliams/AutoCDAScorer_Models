import numpy as np
import pandas as pd
import pytest

from soft_labels import build_soft_targets, summarise


def write_csvs(tmp_path, scores, labels=None, extra_crop=False):
    """Write a manifest and a rater file for one CDA per row of ``scores``."""
    raters = pd.DataFrame(
        {
            "Basename": [f"DSC_{i:04d}.TIF" for i in range(len(scores))],
            "Row": 1,
            "Col": 2,
            "Pos": 3,
            "Score1": [s[0] for s in scores],
            "Score2": [s[1] for s in scores],
            "Score3": [s[2] for s in scores],
            "Median_Score": [np.nanmedian(s) for s in scores],
        }
    )
    if labels is None:
        labels = raters["Median_Score"].tolist()
    files = [f"DSC_{i:04d}_1_2_3.tif" for i in range(len(scores))]
    splits = ["train", "val"] * len(scores)
    if extra_crop:
        files.append("DSC_9999_1_2_3.tif")
        labels = [*labels, 0]
    manifest = pd.DataFrame(
        {
            "split": splits[: len(files)],
            "filename": files,
            "full_path": files,
            "label": labels,
        }
    )
    manifest_csv, rater_csv = tmp_path / "manifest.csv", tmp_path / "raters.csv"
    manifest.to_csv(manifest_csv, index=False)
    raters.to_csv(rater_csv, index=False)
    return manifest_csv, rater_csv


class TestBuildSoftTargets:
    def test_targets_are_rater_proportions(self, tmp_path):
        manifest_csv, rater_csv = write_csvs(tmp_path, [(3, 4, 4), (0, 0, 0), (1, 2, 3)])
        targets, merged = build_soft_targets(manifest_csv, rater_csv)
        np.testing.assert_allclose(targets[0], [0, 0, 0, 1 / 3, 2 / 3, 0, 0], rtol=1e-6)
        np.testing.assert_allclose(targets[1], [1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(targets[2], [0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0], rtol=1e-6)
        assert targets.dtype == np.float32
        assert len(merged) == 3

    def test_missing_score_is_left_out(self, tmp_path, capsys):
        manifest_csv, rater_csv = write_csvs(tmp_path, [(0, 0, np.nan), (5, 5, 5)])
        targets, _ = build_soft_targets(manifest_csv, rater_csv)
        np.testing.assert_allclose(targets[0], [1, 0, 0, 0, 0, 0, 0])
        assert "1 CDAs were scored by fewer than 3 raters" in capsys.readouterr().out

    def test_split_filter_keeps_order(self, tmp_path):
        manifest_csv, rater_csv = write_csvs(tmp_path, [(0, 0, 0), (6, 6, 6), (2, 2, 2)])
        targets, merged = build_soft_targets(manifest_csv, rater_csv, split="train")
        assert merged["filename"].tolist() == ["DSC_0000_1_2_3.tif", "DSC_0002_1_2_3.tif"]
        assert targets.argmax(axis=1).tolist() == [0, 2]

    def test_unmatched_crop_raises(self, tmp_path):
        manifest_csv, rater_csv = write_csvs(tmp_path, [(0, 0, 0)], extra_crop=True)
        with pytest.raises(ValueError, match="no rater record"):
            build_soft_targets(manifest_csv, rater_csv)

    def test_label_mismatch_raises(self, tmp_path):
        manifest_csv, rater_csv = write_csvs(tmp_path, [(2, 2, 2)], labels=[3])
        with pytest.raises(ValueError, match="join is wrong"):
            build_soft_targets(manifest_csv, rater_csv)

    def test_cda_with_no_scores_raises(self, tmp_path):
        manifest_csv, rater_csv = write_csvs(tmp_path, [(np.nan, np.nan, np.nan)], labels=[0])
        raters = pd.read_csv(rater_csv)
        raters["Median_Score"] = 0.0
        raters.to_csv(rater_csv, index=False)
        with pytest.raises(ValueError, match="no scorer values"):
            build_soft_targets(manifest_csv, rater_csv)


class TestSummarise:
    def test_counts_agreement(self):
        targets = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1 / 3, 2 / 3, 0, 0, 0, 0],
                [1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        s = summarise(targets)
        assert s["n"] == 3
        assert s["unanimous (all 3 agree)"] == 1
        assert s["two of three agree"] == 1
        assert s["all three differ"] == 1
        assert s["mean peak mass"] == pytest.approx((1 + 2 / 3 + 1 / 3) / 3)
        expected_entropy = (0 - (1 / 3 * np.log(1 / 3) + 2 / 3 * np.log(2 / 3)) - np.log(1 / 3)) / 3
        assert s["mean entropy (nats)"] == pytest.approx(expected_entropy, rel=1e-5)
