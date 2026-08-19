"""
Test dei descrittori di tessitura cromatinica (Fase 3, STEP 4).

Decisione D2 del piano: GLCM, LBP e statistiche di intensita' si calcolano sui
soli pixel nucleari, non sull'intera patch. Il titolo del lavoro e il report
2.6 parlano di tessitura *cromatinica*: includere stroma e spazio inter-nucleare
diluirebbe il segnale che si vuole quantificare.

Decisione D3: GLCM quantizzata a 64 livelli, distanza 1 px, 4 angoli mediati.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

TEXTURE_COLUMNS = {
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "lbp_entropy",
    "hchannel_mean",
    "hchannel_std",
}


@pytest.fixture(scope="module")
def features():
    spec = importlib.util.spec_from_file_location(
        "mod_features", SRC_DIR / "03_feature_extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _centred_mask(shape=(32, 32)):
    mask = np.zeros(shape, dtype=np.int32)
    mask[8:24, 8:24] = 1
    return mask


def test_returns_exactly_the_six_documented_columns(features):
    h_channel = np.full((32, 32), 120, dtype=np.uint8)

    result = features.extract_texture_features(h_channel, _centred_mask())

    assert set(result) == TEXTURE_COLUMNS


def test_a_uniform_nuclear_region_has_zero_contrast_and_maximal_homogeneity(features):
    h_channel = np.full((32, 32), 120, dtype=np.uint8)

    result = features.extract_texture_features(h_channel, _centred_mask())

    assert result["glcm_contrast"] == pytest.approx(0.0)
    assert result["glcm_homogeneity"] == pytest.approx(1.0)
    assert result["hchannel_mean"] == pytest.approx(120.0)
    assert result["hchannel_std"] == pytest.approx(0.0)


def test_intensity_statistics_ignore_the_background(features):
    """Regressione D2: uno sfondo scuro non deve abbassare la media nucleare."""
    h_channel = np.zeros((32, 32), dtype=np.uint8)
    h_channel[8:24, 8:24] = 200

    result = features.extract_texture_features(h_channel, _centred_mask())

    assert result["hchannel_mean"] == pytest.approx(200.0)
    assert result["hchannel_std"] == pytest.approx(0.0)


def test_the_background_does_not_contribute_to_the_glcm(features):
    """La tessitura di una regione uniforme resta piatta anche con sfondo variabile.

    Se le coppie di pixel che toccano lo sfondo non venissero scartate, il salto
    di intensita' al bordo del nucleo produrrebbe contrasto.
    """
    mask = _centred_mask()
    rng = np.random.default_rng(0)
    h_channel = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    h_channel[mask > 0] = 120  # nucleo uniforme dentro uno sfondo rumoroso

    result = features.extract_texture_features(h_channel, mask)

    assert result["glcm_contrast"] == pytest.approx(0.0)
    assert result["glcm_homogeneity"] == pytest.approx(1.0)


def test_a_noisy_region_has_higher_contrast_than_a_uniform_one(features):
    mask = np.ones((32, 32), dtype=np.int32)
    uniform = np.full((32, 32), 120, dtype=np.uint8)
    checkerboard = np.indices((32, 32)).sum(axis=0) % 2
    noisy = np.where(checkerboard == 0, 40, 200).astype(np.uint8)

    uniform_result = features.extract_texture_features(uniform, mask)
    noisy_result = features.extract_texture_features(noisy, mask)

    assert noisy_result["glcm_contrast"] > uniform_result["glcm_contrast"]
    assert noisy_result["glcm_homogeneity"] < uniform_result["glcm_homogeneity"]


def test_a_uniform_region_has_zero_lbp_entropy(features):
    """Un'unica configurazione locale ricorrente non porta informazione."""
    h_channel = np.full((32, 32), 120, dtype=np.uint8)

    result = features.extract_texture_features(h_channel, _centred_mask())

    assert result["lbp_entropy"] == pytest.approx(0.0)


def test_a_random_region_has_higher_lbp_entropy_than_a_uniform_one(features):
    mask = np.ones((64, 64), dtype=np.int32)
    rng = np.random.default_rng(1)
    uniform = np.full((64, 64), 120, dtype=np.uint8)
    random_texture = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)

    uniform_result = features.extract_texture_features(uniform, mask)
    random_result = features.extract_texture_features(random_texture, mask)

    assert random_result["lbp_entropy"] > uniform_result["lbp_entropy"]


def test_every_column_is_nan_when_the_mask_has_no_nuclei(features):
    h_channel = np.full((32, 32), 120, dtype=np.uint8)
    empty_mask = np.zeros((32, 32), dtype=np.int32)

    result = features.extract_texture_features(h_channel, empty_mask)

    assert set(result) == TEXTURE_COLUMNS
    assert all(math.isnan(v) for v in result.values())


def test_texture_is_nan_when_no_two_nuclear_pixels_are_adjacent(features):
    """Nuclei ridotti a pixel isolati: la GLCM mascherata resta vuota.

    Senza guardia, graycoprops restituirebbe zeri indistinguibili da una
    tessitura perfettamente piatta.
    """
    h_channel = np.full((32, 32), 120, dtype=np.uint8)
    sparse_mask = np.zeros((32, 32), dtype=np.int32)
    sparse_mask[::4, ::4] = 1  # pixel isolati, mai adiacenti a distanza 1

    result = features.extract_texture_features(h_channel, sparse_mask)

    assert math.isnan(result["glcm_contrast"])
    assert math.isnan(result["glcm_homogeneity"])
    assert math.isnan(result["glcm_energy"])
    # Le statistiche di intensita' restano calcolabili: non richiedono adiacenza.
    assert result["hchannel_mean"] == pytest.approx(120.0)


def test_homogeneity_and_energy_stay_within_their_theoretical_range(features):
    rng = np.random.default_rng(0)
    h_channel = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.int32)

    result = features.extract_texture_features(h_channel, mask)

    assert 0.0 < result["glcm_homogeneity"] <= 1.0
    assert 0.0 < result["glcm_energy"] <= 1.0
    assert result["glcm_contrast"] >= 0.0
    assert result["lbp_entropy"] >= 0.0


def test_the_documented_parameters_are_the_approved_ones(features):
    """D3: 64 livelli, distanza 1 px, 4 angoli."""
    assert features.GLCM_LEVELS == 64
    assert tuple(features.GLCM_DISTANCES) == (1,)
    assert tuple(features.GLCM_ANGLES_DEG) == (0, 45, 90, 135)
    assert features.LBP_METHOD == "uniform"
