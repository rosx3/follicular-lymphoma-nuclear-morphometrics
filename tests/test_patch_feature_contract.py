"""
Test del contratto delle colonne prodotte per patch (Fase 3).

Il CSV per patch e' l'input della Fase 4: la sua forma (600 righe x 50 colonne,
47 feature + 3 metadati) e' dichiarata in reports/fase3_report.md 2. Senza un
contratto esplicito, una feature dimenticata o rinominata produrrebbe un CSV
diverso dal previsto senza alcun errore, e il problema emergerebbe solo a valle.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture(scope="module")
def features():
    spec = importlib.util.spec_from_file_location(
        "mod_features", SRC_DIR / "03_feature_extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def four_nuclei_patch():
    """Patch sintetica con 4 nuclei: sufficiente perche' anche k=3 sia definito.

    Le dimensioni sono volutamente diverse fra loro: con nuclei identici la
    skewness e' calcolata su una distribuzione degenere e scipy segnala una
    perdita di precisione.
    """
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:20, 10:20] = 1   # 10x10
    mask[10:22, 30:40] = 2   # 12x10
    mask[30:40, 10:24] = 3   # 10x14
    mask[30:38, 30:38] = 4   # 8x8
    rng = np.random.default_rng(0)
    h_channel = rng.integers(80, 200, size=(64, 64), dtype=np.uint8)
    return h_channel, mask


def test_the_contract_declares_47_features_and_3_metadata(features):
    assert len(features.PATCH_FEATURE_COLUMNS) == 47
    assert len(features.PATCH_METADATA_COLUMNS) == 3


def test_the_contract_has_no_duplicate_column_names(features):
    columns = list(features.PATCH_METADATA_COLUMNS) + list(features.PATCH_FEATURE_COLUMNS)
    assert len(set(columns)) == len(columns)


def test_the_contract_matches_the_sections_of_the_report(features):
    """3 densita + 2 Iwamoto + 32 morfometria + 4 k-NN + 6 tessitura = 47."""
    columns = features.PATCH_FEATURE_COLUMNS

    assert sum(1 for c in columns if c.startswith("knn")) == 4
    assert sum(1 for c in columns if c.startswith(("glcm_", "lbp_", "hchannel_"))) == 6
    assert sum(1 for c in columns if c.startswith("area_top10")) == 2
    morphometry = [
        c for c in columns
        if any(c == f"{base}_{stat}" for base in features.MORPHOMETRY_BASE_FEATURES
               for stat in ("mean", "std", "skew", "cv"))
    ]
    assert len(morphometry) == 32


def test_a_full_patch_row_produces_exactly_the_declared_columns(features, four_nuclei_patch):
    h_channel, mask = four_nuclei_patch

    nuclei = features.extract_nucleus_morphometry(mask)
    row = features.aggregate_patch_morphometry(nuclei, "patch_test", "follicular_lymphoma")
    row.update(features.compute_knn_spatial_features(nuclei))
    row.update(features.extract_texture_features(h_channel, mask))

    produced = set(row) - set(features.PATCH_METADATA_COLUMNS)
    assert produced == set(features.PATCH_FEATURE_COLUMNS)


def test_an_empty_patch_still_produces_every_declared_column(features):
    """Una patch senza nuclei non deve generare una riga con colonne mancanti."""
    empty_mask = np.zeros((64, 64), dtype=np.int32)
    h_channel = np.full((64, 64), 120, dtype=np.uint8)

    nuclei = features.extract_nucleus_morphometry(empty_mask)
    row = features.aggregate_patch_morphometry(nuclei, "patch_vuota", "reactive_tissue")
    row.update(features.compute_knn_spatial_features(nuclei))
    row.update(features.extract_texture_features(h_channel, empty_mask))

    produced = set(row) - set(features.PATCH_METADATA_COLUMNS)
    assert produced == set(features.PATCH_FEATURE_COLUMNS)


def test_the_metadata_columns_come_first_in_the_declared_order(features):
    assert features.PATCH_METADATA_COLUMNS == ("image_name", "category", "target")


def test_skewness_of_a_constant_feature_is_zero_without_warnings(features, four_nuclei_patch):
    """Nuclei perfettamente convessi hanno tutti solidity 1.0.

    Su una distribuzione degenere scipy.stats.skew emette un RuntimeWarning di
    perdita di precisione e restituisce un valore inaffidabile. La skewness di
    una costante e' 0 per definizione e va gestita senza chiamare scipy.
    """
    import warnings

    _h_channel, mask = four_nuclei_patch
    nuclei = features.extract_nucleus_morphometry(mask)
    assert len({n["solidity"] for n in nuclei}) == 1, "il fixture non e' degenere come atteso"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        row = features.aggregate_patch_morphometry(nuclei, "patch_test", "follicular_lymphoma")

    assert row["solidity_skew"] == 0.0
    assert row["solidity_cv"] == 0.0
