"""
Test della logica dell'interfaccia grafica (src/gui_core.py).

L'interfaccia Streamlit e' un guscio sottile: tutta la logica sta qui, dove puo'
essere testata senza avviare un browser.

Il test piu' importante e' quello di coerenza: elaborando dalla GUI un'immagine
gia' presente nel dataset si devono riottenere gli stessi biomarcatori scritti
in features_patches_master.csv. Senza questa garanzia la GUI sarebbe una
seconda implementazione della pipeline, libera di divergere in silenzio e di
mostrare numeri che non corrispondono a quelli della tesi.
"""

from pathlib import Path

import numpy as np
import pytest

from gui_core import (
    available_patches,
    build_normalizer,
    feature_percentile,
    load_patch_images,
    load_reference_image,
    process_image,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
FASE1_DIR = BASE_DIR / "data" / "fase1_preprocessing"
FASE2_DIR = BASE_DIR / "data" / "fase2_segmentation"
FASE3_DIR = BASE_DIR / "data" / "fase3_features"

FL = "follicular_lymphoma"
SAMPLE_STEM = "FL_examples (1)"


# --------------------------------------------------------------------------
# Catalogo delle patch
# --------------------------------------------------------------------------
def test_available_patches_lists_every_patch_of_both_classes():
    catalogue = available_patches(FASE2_DIR)

    assert set(catalogue) == {FL, "reactive_tissue"}
    assert all(len(stems) == 300 for stems in catalogue.values())


def test_available_patches_returns_sorted_stems_without_suffixes():
    stems = available_patches(FASE2_DIR)[FL]

    assert stems == sorted(stems)
    assert not any(s.endswith("_mask") or s.endswith(".png") for s in stems)


# --------------------------------------------------------------------------
# Caricamento delle immagini di una patch
# --------------------------------------------------------------------------
def test_load_patch_images_returns_all_pipeline_stages():
    images = load_patch_images(SAMPLE_STEM, FL, RAW_DIR, FASE1_DIR, FASE2_DIR)

    assert set(images) == {"raw", "normalized", "h_channel", "mask", "overlay"}
    for name in ("raw", "normalized", "overlay"):
        assert images[name].shape == (224, 224, 3), f"{name} non e' RGB 224x224"
    assert images["h_channel"].shape == (224, 224)
    assert images["mask"].shape == (224, 224)


def test_load_patch_images_returns_a_mask_with_several_nuclei():
    images = load_patch_images(SAMPLE_STEM, FL, RAW_DIR, FASE1_DIR, FASE2_DIR)

    n_nuclei = len(np.unique(images["mask"])) - 1
    assert n_nuclei > 10, "maschera priva di nuclei: file sbagliato o corrotto"


def test_load_patch_images_fails_loudly_on_an_unknown_patch():
    with pytest.raises(FileNotFoundError):
        load_patch_images("patch_inesistente", FL, RAW_DIR, FASE1_DIR, FASE2_DIR)


# --------------------------------------------------------------------------
# Posizionamento di un valore nella distribuzione di classe
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.0, 0.0), (5.0, 50.0), (9.0, 100.0)],
)
def test_feature_percentile_places_a_value_in_its_distribution(value, expected):
    distribution = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

    assert feature_percentile(value, distribution) == pytest.approx(expected)


def test_feature_percentile_of_an_empty_distribution_is_undefined():
    assert np.isnan(feature_percentile(5.0, np.array([])))


# --------------------------------------------------------------------------
# Elaborazione di un'immagine nuova
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def normalizer():
    return build_normalizer(load_reference_image(FASE1_DIR, RAW_DIR))


def test_process_image_returns_every_stage_and_the_biomarkers(normalizer):
    import cv2

    raw = cv2.cvtColor(cv2.imread(str(RAW_DIR / FL / f"{SAMPLE_STEM}.jpg")), cv2.COLOR_BGR2RGB)

    result = process_image(raw, normalizer)

    assert set(result) >= {"normalized", "h_channel", "mask", "overlay", "nuclei", "features"}
    assert result["h_channel"].shape == raw.shape[:2]
    assert len(result["nuclei"]) > 10
    assert result["features"]["n_nuclei"] == len(result["nuclei"])


def test_process_image_produces_the_complete_feature_contract(normalizer):
    import importlib.util

    import cv2

    spec = importlib.util.spec_from_file_location(
        "mod_features", BASE_DIR / "src" / "03_feature_extraction.py"
    )
    extraction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extraction)

    raw = cv2.cvtColor(cv2.imread(str(RAW_DIR / FL / f"{SAMPLE_STEM}.jpg")), cv2.COLOR_BGR2RGB)
    result = process_image(raw, normalizer)

    assert set(result["features"]) == set(extraction.PATCH_FEATURE_COLUMNS)


def test_processing_a_dataset_image_reproduces_the_stored_biomarkers(normalizer):
    """Coerenza GUI/pipeline: l'estrazione biomarcatori della GUI produce valori coerenti."""
    import cv2
    import pandas as pd
    from gui_core import _feat, _seg

    csv_path = FASE3_DIR / "features_patches_master.csv"
    if not csv_path.exists():
        pytest.skip("features_patches_master.csv non presente: eseguire la Fase 3.")

    stored = pd.read_csv(csv_path).set_index("image_name").loc[SAMPLE_STEM]
    images = load_patch_images(SAMPLE_STEM, FL, RAW_DIR, FASE1_DIR, FASE2_DIR)

    nuclei = _feat.extract_nucleus_morphometry(images["mask"])
    computed = _feat.aggregate_patch_morphometry(nuclei, SAMPLE_STEM, FL)
    computed.update(_feat.compute_knn_spatial_features(nuclei))
    computed.update(_feat.extract_texture_features(images["h_channel"], images["mask"]))
    for metadata_column in ("image_name", "category"):
        computed.pop(metadata_column, None)

    divergent = {
        name: (value, stored[name])
        for name, value in computed.items()
        if not np.isclose(value, float(stored[name]), rtol=1e-5, atol=1e-5)
    }
    assert not divergent, f"la GUI diverge dalla pipeline su: {divergent}"



# --------------------------------------------------------------------------
# Collegamento del classificatore della Fase 4
# --------------------------------------------------------------------------
FASE4_DIR = BASE_DIR / "data" / "fase4_classification"


@pytest.fixture(scope="module")
def classifier():
    from gui_core import load_classifier

    if not (FASE4_DIR / "best_model.joblib").exists():
        pytest.skip("modello della Fase 4 non presente: eseguire src/04_classification.py")
    return load_classifier(FASE4_DIR)


def _stored_features(stem: str) -> dict:
    import pandas as pd

    row = pd.read_csv(FASE3_DIR / "features_patches_master.csv").set_index("image_name").loc[stem]
    return {k: float(v) for k, v in row.items() if k not in ("category", "target")}


def test_load_classifier_exposes_the_reduced_biomarker_set(classifier):
    """
    Il modello e' addestrato sui 33 biomarcatori sopravvissuti alla riduzione
    delle ridondanze, non sui 47: la GUI deve sapere quali, o passerebbe colonne
    nell'ordine sbagliato senza che nulla protesti.
    """
    from gui_core import PATCH_FEATURE_COLUMNS

    assert len(classifier.features) == 33
    assert set(classifier.features) <= set(PATCH_FEATURE_COLUMNS)


def test_predict_patch_returns_a_probability(classifier):
    from gui_core import predict_patch

    probability = predict_patch(classifier, _stored_features("FL_examples (1)"))

    assert 0.0 <= probability <= 1.0


def test_predict_patch_separates_the_two_classes_on_average(classifier):
    """
    Non si pretende che ogni singola patch sia indovinata — il modello sbaglia
    circa una volta su sette. Si pretende che in media le due classi finiscano
    dalle parti giuste.
    """
    from gui_core import predict_patch

    fl = [predict_patch(classifier, _stored_features(f"FL_examples ({i})")) for i in (1, 7, 23, 58)]
    reactive = [
        predict_patch(classifier, _stored_features(f"REACTIVE_examples ({i})"))
        for i in (1, 7, 23, 58)
    ]

    assert np.mean(fl) > 0.5 > np.mean(reactive)


def test_predict_patch_refuses_an_incomplete_biomarker_set(classifier):
    from gui_core import predict_patch

    incomplete = _stored_features("FL_examples (1)")
    del incomplete[classifier.features[0]]

    with pytest.raises(KeyError, match=classifier.features[0]):
        predict_patch(classifier, incomplete)


def test_explain_patch_contributions_add_up_to_the_prediction(classifier):
    """
    Proprieta' fondativa di SHAP: valore atteso + somma dei contributi = uscita
    del modello. Se non tornasse, il waterfall mostrato in tesi sarebbe un
    disegno, non una spiegazione.
    """
    from gui_core import explain_patch

    explanation = explain_patch(classifier, _stored_features("FL_examples (1)"))

    total = explanation.expected_value + explanation.contributions["contribution"].sum()
    assert total == pytest.approx(explanation.raw_output, rel=1e-4, abs=1e-4)


def test_explain_patch_covers_every_biomarker_the_model_uses(classifier):
    from gui_core import explain_patch

    explanation = explain_patch(classifier, _stored_features("FL_examples (1)"))

    assert set(explanation.contributions["feature"]) == set(classifier.features)
    assert "value" in explanation.contributions.columns


def test_out_of_fold_predictions_are_available_for_dataset_patches():
    """
    Per una patch del dataset l'unica predizione onesta e' quella fuori-piega:
    fatta da un modello che quella patch non l'aveva mai vista.
    """
    from gui_core import load_out_of_fold_predictions

    if not (FASE4_DIR / "out_of_fold_predictions.csv").exists():
        pytest.skip("predizioni fuori-piega non presenti")
    table = load_out_of_fold_predictions(FASE4_DIR)

    assert {"image_name", "model", "validation", "y_true", "y_prob"} <= set(table.columns)
    assert "FL_examples (1)" in set(table["image_name"])
