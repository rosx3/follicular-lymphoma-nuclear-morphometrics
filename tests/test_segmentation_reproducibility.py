"""
Riproducibilita' della Fase 2: i default del modulo devono rigenerare le
maschere del dataset.

MOTIVAZIONE. Le 600 maschere in data/fase2_segmentation/ sono la base di tutta
la Fase 3 e quindi di ogni numero della tesi, ma furono prodotte (agosto 2026,
commit 9c59248) da un runner esterno al repository: run_pipeline.py non esisteva
ancora. I parametri di quel run non erano registrati da nessuna parte, e i
default rimasti nel codice non erano gli stessi: `python src/run_pipeline.py`
rieseguito sulla Fase 2 produceva il 54% di nuclei in meno (media 74.6 contro
163.6 per patch), riscrivendo l'intera matrice dei biomarcatori.

I parametri originali sono stati ricostruiti per ricerca esaustiva su griglia,
verificando l'accordo pixel per pixel con le maschere salvate:
min_distance = 7 px, min_area_px = 15. Questo test impedisce che si perdano di
nuovo: se qualcuno cambia un default della segmentazione, il dataset della tesi
smette di essere riproducibile e il test lo dice subito.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
FASE1_DIR = BASE_DIR / "data" / "fase1_preprocessing"
FASE2_DIR = BASE_DIR / "data" / "fase2_segmentation"

# Un campione per classe: il test deve restare rapido, la ricostruzione e' stata
# validata su 60 patch.
SAMPLE_PATCHES = [
    ("follicular_lymphoma", "FL_examples (1)"),
    ("follicular_lymphoma", "FL_examples (23)"),
    ("reactive_tissue", "REACTIVE_examples (1)"),
    ("reactive_tissue", "REACTIVE_examples (23)"),
]


@pytest.fixture(scope="module")
def segmentation():
    import importlib.util
    import sys

    src_dir = BASE_DIR / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    spec = importlib.util.spec_from_file_location(
        "segmentation_under_test", src_dir / "02_segmentation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stored(category: str, stem: str) -> tuple[np.ndarray, np.ndarray]:
    h_channel = cv2.imread(
        str(FASE1_DIR / category / "h_channel" / f"{stem}_hchannel.png"), cv2.IMREAD_GRAYSCALE
    )
    mask = cv2.imread(
        str(FASE2_DIR / category / "masks" / f"{stem}_mask.png"), cv2.IMREAD_UNCHANGED
    )
    assert h_channel is not None and mask is not None, f"input mancante per {stem}"
    return h_channel, mask


@pytest.mark.parametrize(("category", "stem"), SAMPLE_PATCHES)
def test_default_parameters_reproduce_the_stored_mask(segmentation, category, stem):
    h_channel, stored_mask = _stored(category, stem)

    computed_mask, _ = segmentation.segment_nuclei_watershed(h_channel)

    divergent_pixels = int(((stored_mask > 0) != (computed_mask > 0)).sum())
    assert divergent_pixels == 0, (
        f"{stem}: {divergent_pixels} pixel divergono dalla maschera del dataset — "
        "i default della segmentazione non riproducono piu' la Fase 2."
    )


@pytest.mark.parametrize(("category", "stem"), SAMPLE_PATCHES)
def test_default_parameters_reproduce_the_stored_nucleus_count(segmentation, category, stem):
    h_channel, stored_mask = _stored(category, stem)

    computed_mask, centroids = segmentation.segment_nuclei_watershed(h_channel)

    stored_count = len(np.unique(stored_mask)) - 1
    assert len(np.unique(computed_mask)) - 1 == stored_count
    assert len(centroids) == stored_count
