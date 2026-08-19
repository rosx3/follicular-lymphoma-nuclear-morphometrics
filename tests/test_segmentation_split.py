"""
Test dello split stratificato train/val della Fase 2.

Contesto (problema B2 dell'audit Fase 3): split_gt_patches
selezionava le due classi confrontando la categoria con le stringhe letterali
'Follicular Lymphoma' / 'Reactive Tissue'. Con le etichette canoniche
('follicular_lymphoma' / 'reactive_tissue') entrambi i gruppi risultavano
vuoti e la funzione restituiva due liste vuote senza alcun errore.
"""

import importlib.util
from pathlib import Path

import pytest

from naming import CATEGORY_FL, CATEGORY_REACTIVE

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture(scope="module")
def segmentation():
    spec = importlib.util.spec_from_file_location("mod_segmentation", SRC_DIR / "02_segmentation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patches(category, n):
    return [(f"{category}_{i}.png", f"{category}_{i}_mask.png", category) for i in range(n)]


@pytest.mark.parametrize(
    ("fl_label", "re_label"),
    [
        (CATEGORY_FL, CATEGORY_REACTIVE),          # convenzione canonica
        ("Follicular Lymphoma", "Reactive Tissue"),  # convenzione storica
        ("FL", "REACTIVE"),                          # etichette brevi
    ],
)
def test_split_stratifies_every_category_convention(segmentation, fl_label, re_label):
    patches = _patches(fl_label, 6) + _patches(re_label, 6)

    train, val = segmentation.split_gt_patches(patches, val_fraction=0.5, seed=0)

    assert len(train) + len(val) == 12, "alcune patch sono state perse dallo split"
    for subset in (train, val):
        assert sum(1 for p in subset if p[2] == fl_label) == 3
        assert sum(1 for p in subset if p[2] == re_label) == 3


def test_split_rejects_an_unknown_category(segmentation):
    with pytest.raises(ValueError):
        segmentation.split_gt_patches(_patches("tessuto_ignoto", 4), val_fraction=0.5)
