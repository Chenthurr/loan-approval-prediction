import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import DataGenerator


def test_generator_is_reproducible_and_has_expected_schema():
    first = DataGenerator.generate(n_samples=32)
    second = DataGenerator.generate(n_samples=32)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 32
    assert "loan_approved" in first.columns
    assert set(first["loan_approved"].unique()).issubset({0, 1})


def test_generator_respects_sample_count():
    dataset = DataGenerator.generate(n_samples=7)

    assert dataset.shape == (7, 10)
