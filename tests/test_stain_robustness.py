"""
Test dell'esperimento di robustezza alla colorazione (src/stain_robustness.py).

La perturbazione deve fare una cosa sola: cambiare COME i nuclei sono colorati,
senza toccare DOVE sono e che forma hanno. Se alterasse la geometria, il test
misurerebbe la robustezza a una degradazione dell'immagine invece che alla
variabilita' di colorazione, e la conclusione sarebbe priva di valore.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
SAMPLE = BASE_DIR / "data" / "raw" / "follicular_lymphoma" / "FL_examples (1).jpg"


@pytest.fixture(scope="module")
def robustness():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    spec = importlib.util.spec_from_file_location(
        "stain_robustness_under_test", SRC_DIR / "stain_robustness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def raw_image():
    import cv2

    if not SAMPLE.exists():
        pytest.skip("patch di prova non presente")
    return cv2.cvtColor(cv2.imread(str(SAMPLE), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def test_perturbation_preserves_shape_and_type(robustness, raw_image):
    perturbed = robustness.perturb_stain(raw_image, 0.2, np.random.default_rng(0))

    assert perturbed.shape == raw_image.shape
    assert perturbed.dtype == raw_image.dtype


def test_zero_perturbation_reproduces_the_image_almost_exactly(robustness, raw_image):
    """
    Con sigma nullo resta il solo ciclo scomposizione/ricomposizione. Un errore
    grande qui vorrebbe dire che parte dello scarto misurato a sigma > 0 non e'
    la perturbazione ma il ciclo stesso: il controllo serve a escluderlo.
    """
    unchanged = robustness.perturb_stain(raw_image, 0.0, np.random.default_rng(0))

    difference = np.abs(unchanged.astype(float) - raw_image.astype(float))
    assert difference.mean() < 12.0, f"il solo ciclo sposta gia' {difference.mean():.1f}/255"


def test_a_stronger_perturbation_changes_the_image_more(robustness, raw_image):
    def distance(sigma, seed):
        perturbed = robustness.perturb_stain(raw_image, sigma, np.random.default_rng(seed))
        return np.abs(perturbed.astype(float) - raw_image.astype(float)).mean()

    mild = np.mean([distance(0.10, s) for s in range(5)])
    strong = np.mean([distance(0.30, s) for s in range(5)])

    assert strong > mild, f"sigma 0.30 ({strong:.1f}) non supera sigma 0.10 ({mild:.1f})"


def test_perturbation_does_not_move_the_nuclei(robustness, raw_image):
    """
    La prova che il test misura la colorazione e non un danno all'immagine:
    segmentando l'immagine perturbata si devono ritrovare all'incirca gli stessi
    nuclei, negli stessi posti.
    """
    import gui_core

    perturbed = robustness.perturb_stain(raw_image, 0.2, np.random.default_rng(1))
    normalizer = gui_core.build_normalizer(
        gui_core.load_reference_image(
            BASE_DIR / "data" / "fase1_preprocessing", BASE_DIR / "data" / "raw"
        )
    )

    before = gui_core.process_image(raw_image, normalizer)["mask"] > 0
    after = gui_core.process_image(perturbed, normalizer)["mask"] > 0

    overlap = np.logical_and(before, after).sum() / np.logical_or(before, after).sum()
    assert overlap > 0.75, f"la geometria dei nuclei e' cambiata troppo (IoU {overlap:.2f})"


def test_a_negative_sigma_is_refused(robustness, raw_image):
    with pytest.raises(ValueError, match="negativo"):
        robustness.perturb_stain(raw_image, -0.1, np.random.default_rng(0))
