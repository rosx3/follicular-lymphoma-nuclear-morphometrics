"""
Test delle distanze micro-spaziali k-NN (Fase 3, STEP 3).

Le distanze ai vicini piu' prossimi sostituiscono Delaunay e MST come
descrittori di micro-architettura del packing nucleare: su patch da 103 um
i grafi spaziali sono compromessi dai boundary effects
(vedi reports/fase3_report.md 1.1 e 7, decisione D7 del piano).

I casi non definiti valgono NaN e non 0.0 (decisione D1): uno zero verrebbe
letto dal modello come densita' massima, cioe' l'opposto della situazione
reale di una patch quasi vuota.
"""

import importlib.util
import math
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

KNN_COLUMNS = {
    "knn1_dist_mean_um",
    "knn1_dist_std_um",
    "knn3_dist_mean_um",
    "knn3_dist_std_um",
}


@pytest.fixture(scope="module")
def features():
    spec = importlib.util.spec_from_file_location(
        "mod_features", SRC_DIR / "03_feature_extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nuclei(*coords):
    """Costruisce la lista di nuclei minimale accettata dalla funzione."""
    return [{"centroid_x_um": x, "centroid_y_um": y} for x, y in coords]


# Quadrato unitario: ogni vertice dista 1.0 dai due adiacenti e sqrt(2)
# dall'opposto. Tutte le attese sono quindi calcolabili a mano.
UNIT_SQUARE = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


def test_knn1_on_a_unit_square_is_the_side_length(features):
    result = features.compute_knn_spatial_features(_nuclei(*UNIT_SQUARE))

    assert result["knn1_dist_mean_um"] == pytest.approx(1.0)
    assert result["knn1_dist_std_um"] == pytest.approx(0.0)


def test_knn3_on_a_unit_square_averages_two_sides_and_one_diagonal(features):
    expected = (1.0 + 1.0 + math.sqrt(2.0)) / 3.0

    result = features.compute_knn_spatial_features(_nuclei(*UNIT_SQUARE))

    assert result["knn3_dist_mean_um"] == pytest.approx(expected, abs=1e-4)
    assert result["knn3_dist_std_um"] == pytest.approx(0.0)


def test_a_nucleus_is_not_its_own_neighbour(features):
    """Se il self-match non fosse escluso, la media conterrebbe uno zero."""
    result = features.compute_knn_spatial_features(_nuclei((0.0, 0.0), (10.0, 0.0)))

    assert result["knn1_dist_mean_um"] == pytest.approx(10.0)


def test_knn3_is_nan_with_only_three_nuclei_but_knn1_is_defined(features):
    result = features.compute_knn_spatial_features(
        _nuclei((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    )

    assert result["knn1_dist_mean_um"] == pytest.approx(1.0)
    assert math.isnan(result["knn3_dist_mean_um"])
    assert math.isnan(result["knn3_dist_std_um"])


@pytest.mark.parametrize("coords", [(), ((0.0, 0.0),)])
def test_every_knn_column_is_nan_when_there_are_fewer_than_two_nuclei(features, coords):
    result = features.compute_knn_spatial_features(_nuclei(*coords))

    assert set(result) == KNN_COLUMNS
    assert all(math.isnan(v) for v in result.values())


def test_returns_exactly_the_four_documented_columns(features):
    result = features.compute_knn_spatial_features(_nuclei(*UNIT_SQUARE))

    assert set(result) == KNN_COLUMNS


def test_distances_scale_with_the_coordinates(features):
    """Controllo dimensionale: raddoppiando le coordinate raddoppiano le distanze."""
    doubled = _nuclei(*[(2 * x, 2 * y) for x, y in UNIT_SQUARE])

    result = features.compute_knn_spatial_features(doubled)

    assert result["knn1_dist_mean_um"] == pytest.approx(2.0)


def test_std_is_non_zero_when_the_packing_is_irregular(features):
    """La std misura la regolarita' del packing: un layout non uniforme la alza."""
    irregular = _nuclei((0.0, 0.0), (1.0, 0.0), (20.0, 0.0), (21.0, 0.0))

    result = features.compute_knn_spatial_features(irregular)

    assert result["knn1_dist_std_um"] == pytest.approx(0.0)
    assert result["knn3_dist_std_um"] > 0.0
