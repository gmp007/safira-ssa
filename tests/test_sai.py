from __future__ import annotations

import unittest

import pandas as pd

from safira.sai import build_composite_sai, get_standard_country_code


class SkillAdvancementIndexTests(unittest.TestCase):
    def test_country_aliases_are_accent_and_punctuation_tolerant(self):
        self.assertEqual(get_standard_country_code("Côte d’Ivoire"), "CIV")
        self.assertEqual(get_standard_country_code("South Africa"), "ZAF")
        self.assertEqual(get_standard_country_code("DRC"), "COD")

    def test_build_composite_sai_outputs_expected_columns_and_bounds(self):
        df = pd.DataFrame(
            {
                "year": [2000, 2001, 2000, 2001],
                "country": ["Nigeria", "Nigeria", "Kenya", "Kenya"],
                "Literacy_Rate_Adult_Total": [50, 55, 60, 65],
                "Literacy_Rate_Youth_Total": [55, 60, 65, 70],
                "Primary_Completion_Rate_Total": [45, 50, 55, 60],
                "Enrollment_Tertiary": [7, 8, 9, 10],
                "R_and_D_Expenditure": [0.1, 0.2, 0.3, 0.4],
                "Secure_Internet_Servers": [10, 20, 30, 40],
                "Fixed_Broadband_Subscriptions": [0.5, 1.0, 1.5, 2.0],
                "Unemployment_Total": [12, 10, 8, 6],
            }
        )

        result = build_composite_sai(df)

        for column in ["dim_foundational", "dim_advanced", "dim_digital", "dim_labor", "SAI"]:
            self.assertIn(column, result.columns)
            self.assertTrue(result[column].dropna().between(0, 100).all())


if __name__ == "__main__":
    unittest.main()
