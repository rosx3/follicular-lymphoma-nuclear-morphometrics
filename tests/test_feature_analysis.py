"""
Test dell'analisi di separabilita' statistica FL vs REACTIVE (Fase 3, Task 6).

Decisione D4 del piano: con 47 feature testate simultaneamente a alpha=0.05 ci
si attendono circa 2 falsi positivi per puro caso, quindi si riporta il p-value
grezzo affiancato a quello corretto con Benjamini-Hochberg.

La scelta del test non e' fissata a priori: si verifica la normalita' dei due
gruppi e si usa il t-test di Welch se entrambi la soddisfano, Mann-Whitney U
altrimenti.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_analysis import METADATA_COLUMNS, describe_by_class, separability_tests

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

FL = "follicular_lymphoma"
REACTIVE = "reactive_tissue"


def _frame(**columns):
    """Costruisce un DataFrame con 60 patch per classe e le feature indicate."""
    n = 60
    data = {
        "image_name": [f"p{i}" for i in range(2 * n)],
        "category": [FL] * n + [REACTIVE] * n,
        "target": [1] * n + [0] * n,
    }
    data.update(columns)
    return pd.DataFrame(data)


@pytest.fixture
def separated_frame():
    """Una feature nettamente separata e una indistinguibile."""
    rng = np.random.default_rng(0)
    n = 60
    return _frame(
        molto_separata=np.concatenate([rng.normal(10, 1, n), rng.normal(20, 1, n)]),
        indistinguibile=np.concatenate([rng.normal(5, 1, n), rng.normal(5, 1, n)]),
    )


def test_returns_one_row_per_feature(separated_frame):
    result = separability_tests(separated_frame)

    assert len(result) == 2
    assert set(result["feature"]) == {"molto_separata", "indistinguibile"}


def test_a_clearly_separated_feature_is_significant_with_a_large_effect(separated_frame):
    result = separability_tests(separated_frame).set_index("feature")

    row = result.loc["molto_separata"]
    assert row["p_fdr"] < 0.001
    assert bool(row["significant"]) is True
    assert abs(row["effect_size"]) > 0.8, "effect size non 'grande' secondo Cohen"


def test_two_identical_distributions_are_not_significant(separated_frame):
    result = separability_tests(separated_frame).set_index("feature")

    row = result.loc["indistinguibile"]
    assert row["p_raw"] > 0.05
    assert bool(row["significant"]) is False


def test_the_fdr_correction_never_lowers_a_p_value(separated_frame):
    result = separability_tests(separated_frame)

    assert (result["p_fdr"] >= result["p_raw"] - 1e-12).all()
    assert (result["p_fdr"] <= 1.0).all()


def test_the_effect_size_sign_says_which_class_is_larger():
    rng = np.random.default_rng(1)
    n = 60
    frame = _frame(
        fl_maggiore=np.concatenate([rng.normal(20, 1, n), rng.normal(10, 1, n)]),
        fl_minore=np.concatenate([rng.normal(10, 1, n), rng.normal(20, 1, n)]),
    )

    result = separability_tests(frame).set_index("feature")

    assert result.loc["fl_maggiore", "effect_size"] > 0
    assert result.loc["fl_minore", "effect_size"] < 0


def test_normal_data_uses_welch_and_skewed_data_uses_mann_whitney():
    rng = np.random.default_rng(2)
    n = 60
    frame = _frame(
        gaussiana=np.concatenate([rng.normal(10, 1, n), rng.normal(11, 1, n)]),
        molto_asimmetrica=np.concatenate([rng.exponential(1, n), rng.exponential(3, n)]),
    )

    result = separability_tests(frame).set_index("feature")

    assert result.loc["gaussiana", "test"] == "welch_t"
    assert result.loc["molto_asimmetrica", "test"] == "mann_whitney_u"


def test_a_feature_with_missing_values_is_analysed_on_the_valid_ones():
    rng = np.random.default_rng(3)
    n = 60
    values = np.concatenate([rng.normal(10, 1, n), rng.normal(20, 1, n)])
    values[:5] = np.nan
    frame = _frame(con_nan=values)

    result = separability_tests(frame).set_index("feature")

    assert result.loc["con_nan", "n_fl"] == n - 5
    assert result.loc["con_nan", "n_reactive"] == n
    assert not np.isnan(result.loc["con_nan", "p_raw"])


def test_a_constant_feature_does_not_break_the_analysis():
    """Una colonna costante non e' separabile e non deve far esplodere il test."""
    frame = _frame(costante=np.full(120, 3.0))

    result = separability_tests(frame).set_index("feature")

    assert bool(result.loc["costante", "significant"]) is False
    assert result.loc["costante", "effect_size"] == 0.0


def test_the_results_are_sorted_by_evidence_strength(separated_frame):
    result = separability_tests(separated_frame)

    assert result.iloc[0]["feature"] == "molto_separata"


def test_describe_by_class_reports_both_classes(separated_frame):
    described = describe_by_class(separated_frame)

    assert set(described.index) == {"molto_separata", "indistinguibile"}
    for column in ("mean_fl", "std_fl", "mean_reactive", "std_reactive"):
        assert column in described.columns
    assert described.loc["molto_separata", "mean_fl"] == pytest.approx(10, abs=0.5)
    assert described.loc["molto_separata", "mean_reactive"] == pytest.approx(20, abs=0.5)


def test_metadata_columns_match_the_extraction_contract():
    """Le colonne non-feature devono restare allineate a quelle della Fase 3."""
    spec = importlib.util.spec_from_file_location(
        "mod_features", SRC_DIR / "03_feature_extraction.py"
    )
    extraction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extraction)

    assert METADATA_COLUMNS == extraction.PATCH_METADATA_COLUMNS


def test_the_real_dataset_yields_one_row_for_each_of_the_47_features():
    csv_path = Path(__file__).resolve().parent.parent / "data" / "fase3_features" / "features_patches_master.csv"
    if not csv_path.exists():
        pytest.skip("features_patches_master.csv non presente: eseguire la Fase 3.")

    result = separability_tests(pd.read_csv(csv_path))

    assert len(result) == 47
