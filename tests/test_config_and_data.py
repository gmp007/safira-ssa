from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safira.config import cfg_bool, cfg_int, cfg_list, read_config, write_default_input
from safira.data import prepare_panel, write_sample_panel


class ConfigAndDataTests(unittest.TestCase):
    def test_default_input_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "safira.in"
            write_default_input(path)
            config = read_config(path)

            self.assertTrue(cfg_bool(config, "workflow", "train_model"))
            self.assertEqual(cfg_int(config, "model", "lookback"), 5)
            self.assertIn("spaghetti", cfg_list(config, "plots", "selected"))

    def test_sample_panel_can_be_prepared(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.xlsx"
            write_sample_panel(path)
            frame = prepare_panel(path)

            self.assertIn("SAI", frame.columns)
            self.assertIn("country_code", frame.columns)
            self.assertGreaterEqual(frame["country_code"].nunique(), 5)
            self.assertTrue(frame["SAI"].between(0, 100).all())


if __name__ == "__main__":
    unittest.main()
