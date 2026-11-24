import numpy as np
import pandas as pd

from grid_results import deduplicate, load_grid, setting_key, standard_error, tie_band


def grid(rows):
    """Rows of (model_id, num_filters, reg, reg_strength, avg_vaf)."""
    return pd.DataFrame(rows, columns=["model_id", "num_filters", "reg", "reg_strength", "avg_vaf"])


class TestSettingKey:
    def test_reg_strength_ignored_when_reg_absent(self):
        g = grid([(0, 8, np.nan, 1e-4, 0.6), (1, 8, np.nan, 1e-2, 0.6)])
        assert setting_key(g).nunique() == 1

    def test_reg_strength_counts_when_reg_present(self):
        g = grid([(0, 8, "L2", 1e-4, 0.6), (1, 8, "L2", 1e-2, 0.6)])
        assert setting_key(g).nunique() == 2

    def test_other_axes_separate_settings(self):
        g = grid([(0, 8, np.nan, 1e-4, 0.6), (1, 16, np.nan, 1e-4, 0.6)])
        assert setting_key(g).nunique() == 2

    def test_metric_columns_take_no_part(self):
        g = grid([(0, 8, np.nan, 1e-4, 0.60), (1, 8, np.nan, 1e-3, 0.64)])
        assert setting_key(g).nunique() == 1


class TestDeduplicate:
    def test_keeps_the_best_run_of_each_setting(self):
        g = grid(
            [(0, 8, np.nan, 1e-4, 0.60), (1, 8, np.nan, 1e-3, 0.64), (2, 8, np.nan, 1e-2, 0.62)]
        )
        out = deduplicate(g)
        assert list(out.model_id) == [1]

    def test_distinct_settings_all_survive(self):
        g = grid([(0, 8, "L2", 1e-4, 0.60), (1, 16, "L2", 1e-4, 0.64)])
        assert len(deduplicate(g)) == 2

    def test_ties_broken_by_model_id_not_row_order(self):
        rows = [(5, 8, np.nan, 1e-4, 0.6), (2, 8, np.nan, 1e-3, 0.6)]
        assert list(deduplicate(grid(rows)).model_id) == [2]
        assert list(deduplicate(grid(rows[::-1])).model_id) == [2]

    def test_respects_the_named_metric(self):
        g = grid([(0, 8, np.nan, 1e-4, 0.60), (1, 8, np.nan, 1e-3, 0.64)])
        g["avg_vaf_orig"] = [0.75, 0.70]
        assert list(deduplicate(g, "avg_vaf_orig").model_id) == [0]

    def test_no_op_without_a_regulariser_column(self):
        g = grid([(0, 8, "L2", 1e-4, 0.6), (1, 16, "L2", 1e-4, 0.6)]).drop(columns=["reg"])
        assert len(deduplicate(g)) == 2


class TestTieBand:
    def test_keeps_everything_within_one_standard_error(self):
        se = standard_error(0.65)
        g = grid(
            [
                (0, 8, "L2", 1e-4, 0.65),
                (1, 16, "L2", 1e-4, 0.65 - se / 2),
                (2, 32, "L2", 1e-4, 0.65 - se * 2),
            ]
        )
        assert list(tie_band(g).model_id) == [0, 1]

    def test_band_width_follows_the_crop_count(self):
        assert standard_error(0.65, 308) > standard_error(0.65, 2289)

    def test_repeat_runs_would_fill_the_band_undeduplicated(self):
        g = grid(
            [(0, 8, np.nan, 1e-4, 0.650), (1, 8, np.nan, 1e-3, 0.649), (2, 16, "L2", 1e-4, 0.648)]
        )
        assert len(tie_band(g)) == 3
        assert len(tie_band(deduplicate(g))) == 2


class TestLoadGrid:
    def test_merges_unmerged_parts_without_duplicates(self, tmp_path):
        pd.DataFrame({"model_id": [1, 2], "avg_vaf": [0.5, 0.6]}).to_csv(
            tmp_path / "random_search_results.csv", index=False
        )
        (tmp_path / "parts").mkdir()
        pd.DataFrame({"model_id": [2, 3], "avg_vaf": [0.6, 0.7]}).to_csv(
            tmp_path / "parts" / "9.csv", index=False
        )
        assert sorted(load_grid(tmp_path)["model_id"]) == [1, 2, 3]

    def test_without_parts_folder(self, tmp_path):
        pd.DataFrame({"model_id": [1], "avg_vaf": [0.5]}).to_csv(
            tmp_path / "random_search_results.csv", index=False
        )
        assert len(load_grid(tmp_path)) == 1
