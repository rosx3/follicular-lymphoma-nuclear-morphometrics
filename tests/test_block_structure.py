"""
Test della verifica di struttura nell'ordine di numerazione (src/block_structure.py).

Il rapporto misurato da questo modulo e' la sola prova empirica a sostegno della
validazione a blocchi della Fase 4 (spec, decisione D1). Se la misura fosse
sbagliata, l'intera giustificazione della decisione cadrebbe. I test qui
presidiano le due proprieta' che la rendono interpretabile: deve segnalare la
struttura quando c'e', e non deve segnalarla quando non c'e'.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import block_structure as bs  # noqa: E402


def _names(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_examples ({i})" for i in range(1, n + 1)]


def _blocky_matrix(n_blocks: int, per_block: int, n_features: int, seed: int) -> np.ndarray:
    """
    Matrice con struttura a blocchi lungo l'ordine delle righe.

    Ogni blocco ha un proprio centro e le sue righe gli stanno vicino: e'
    l'analogo sintetico di piu' patch estratte dallo stesso vetrino.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 5, size=(n_blocks, n_features))
    return np.vstack([
        centres[b] + rng.normal(0, 0.2, size=(per_block, n_features))
        for b in range(n_blocks)
    ])


def test_structured_order_is_detected():
    X = _blocky_matrix(n_blocks=10, per_block=6, n_features=8, seed=0)

    table = bs.index_order_similarity(X, _names("FL", len(X)), n_permutations=200)

    row = table.iloc[0]
    assert row["ratio"] < 0.7, "una struttura a blocchi netta deve dare un rapporto basso"
    assert row["p_permutation"] < 0.01
    assert row["n_patches"] == 60


def test_shuffled_order_carries_no_structure():
    """La stessa matrice, con le righe mescolate: l'ordine non dice piu' nulla."""
    X = _blocky_matrix(n_blocks=10, per_block=6, n_features=8, seed=0)
    X = X[np.random.default_rng(1).permutation(len(X))]

    table = bs.index_order_similarity(X, _names("FL", len(X)), n_permutations=200)

    row = table.iloc[0]
    assert 0.85 < row["ratio"] < 1.15, "senza struttura il rapporto deve stare intorno a 1"
    assert row["p_permutation"] > 0.05


def test_each_class_is_measured_separately():
    X = np.vstack([
        _blocky_matrix(n_blocks=5, per_block=4, n_features=6, seed=2),
        _blocky_matrix(n_blocks=5, per_block=4, n_features=6, seed=3),
    ])
    names = _names("FL", 20) + _names("REACTIVE", 20)

    table = bs.index_order_similarity(X, names, n_permutations=100)

    assert list(table["prefix"]) == ["FL_examples", "REACTIVE_examples"]
    assert set(table["n_patches"]) == {20}


def test_the_ordering_follows_the_index_not_the_row_order():
    """
    Le righe arrivano in ordine alfabetico, dove "(10)" precede "(2)".

    Se la misura seguisse l'ordine delle righe invece dell'indice numerico,
    leggerebbe come adiacenti coppie che adiacenti non sono, e il rapporto
    perderebbe significato.
    """
    X = _blocky_matrix(n_blocks=6, per_block=5, n_features=6, seed=4)
    indices = sorted(range(1, len(X) + 1), key=str)
    names = [f"FL_examples ({i})" for i in indices]
    # Le righe restano in ordine di indice crescente: e' `names` a essere
    # permutato, quindi la riga i-esima appartiene alla patch indices[i].
    X = X[np.argsort(np.argsort(indices))]

    table = bs.index_order_similarity(X, names, n_permutations=200)

    assert table.iloc[0]["ratio"] < 0.7


def test_a_name_without_an_index_is_rejected():
    X = _blocky_matrix(n_blocks=3, per_block=4, n_features=4, seed=5)
    names = _names("FL", len(X))
    names[3] = "FL_examples_senza_indice"

    with pytest.raises(ValueError, match="indice numerico"):
        bs.index_order_similarity(X, names, n_permutations=10)


def test_a_class_too_small_to_measure_is_rejected():
    X = _blocky_matrix(n_blocks=1, per_block=2, n_features=4, seed=6)

    with pytest.raises(ValueError, match="almeno 3"):
        bs.index_order_similarity(X, _names("FL", len(X)), n_permutations=10)


def test_constant_features_are_dropped_instead_of_producing_nan():
    X = _blocky_matrix(n_blocks=5, per_block=4, n_features=6, seed=7)
    X = np.hstack([X, np.full((len(X), 2), 3.14)])

    table = bs.index_order_similarity(X, _names("FL", len(X)), n_permutations=100)

    assert table.iloc[0]["n_features"] == 6, "le due colonne costanti vanno scartate"
    assert np.isfinite(table.iloc[0]["ratio"])


def test_the_measure_is_reproducible_across_runs():
    X = _blocky_matrix(n_blocks=5, per_block=5, n_features=6, seed=8)
    names = _names("FL", len(X))

    first = bs.index_order_similarity(X, names, n_permutations=100, seed=1)
    second = bs.index_order_similarity(X, names, n_permutations=100, seed=1)

    assert first.equals(second)
