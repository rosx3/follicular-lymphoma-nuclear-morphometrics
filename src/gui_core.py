"""
===============================================================================
gui_core.py — Logica dell'Interfaccia Grafica
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
===============================================================================
Tutta la logica dietro l'interfaccia Streamlit (src/gui.py) vive qui, separata
dai widget, per due ragioni: puo' essere testata senza avviare un browser, e
soprattutto **riusa i moduli della pipeline invece di reimplementarli**.

Una GUI che ricalcola i biomarcatori per conto proprio e' una seconda
implementazione libera di divergere in silenzio, mostrando all'utente numeri
diversi da quelli scritti nei CSV della tesi. Qui ogni passo chiama la stessa
funzione usata da run_pipeline.py, e un test di coerenza verifica che
elaborando un'immagine gia' nel dataset si riottengano i valori memorizzati.
===============================================================================
"""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from naming import (  # noqa: E402
    CATEGORIES,
    h_channel_name,
    mask_name,
    overlay_name,
    rgb_normalized_name,
    stem_from_mask_name,
)


def _load_numbered_module(filename: str, alias: str):
    """Importa i moduli con prefisso numerico, non importabili per nome."""
    spec = importlib.util.spec_from_file_location(alias, _SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prep = _load_numbered_module("01_preprocessing.py", "gui_preprocessing")
_seg = _load_numbered_module("02_segmentation.py", "gui_segmentation")
_feat = _load_numbered_module("03_feature_extraction.py", "gui_features")

PATCH_FEATURE_COLUMNS = _feat.PATCH_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Catalogo delle patch gia' elaborate
# ---------------------------------------------------------------------------
def available_patches(fase2_dir: Path) -> dict[str, list[str]]:
    """Stem delle patch disponibili, per categoria, ordinati alfabeticamente."""
    catalogue = {}
    for category in CATEGORIES:
        mask_dir = Path(fase2_dir) / category / "masks"
        stems = [stem_from_mask_name(p.name) for p in mask_dir.glob("*_mask.png")]
        catalogue[category] = sorted(stems)
    return catalogue


def _imread(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Legge un'immagine sollevando se manca, invece di restituire None."""
    if not path.exists():
        raise FileNotFoundError(f"Immagine non trovata: {path}")
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Immagine illeggibile: {path}")
    return image


def load_patch_images(
    stem: str, category: str, raw_dir: Path, fase1_dir: Path, fase2_dir: Path
) -> dict[str, np.ndarray]:
    """
    Carica tutti gli stadi della pipeline per una patch gia' elaborata.

    Returns:
        dict con 'raw' e 'normalized' (RGB), 'h_channel' e 'mask' (2D),
        'overlay' (RGB con i contorni della segmentazione).
    """
    raw_path = Path(raw_dir) / category / f"{stem}.jpg"
    fase1, fase2 = Path(fase1_dir), Path(fase2_dir)

    return {
        "raw": cv2.cvtColor(_imread(raw_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
        "normalized": cv2.cvtColor(
            _imread(fase1 / category / "rgb_normalized" / rgb_normalized_name(stem), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        ),
        "h_channel": _imread(fase1 / category / "h_channel" / h_channel_name(stem), cv2.IMREAD_GRAYSCALE),
        "mask": _imread(fase2 / category / "masks" / mask_name(stem)),
        "overlay": cv2.cvtColor(
            _imread(fase2 / category / "overlays" / overlay_name(stem), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        ),
    }


# ---------------------------------------------------------------------------
# Posizionamento di un valore nella distribuzione della sua classe
# ---------------------------------------------------------------------------
def feature_percentile(value: float, distribution: np.ndarray) -> float:
    """
    Percentile occupato da `value` nella distribuzione, in [0, 100].

    Serve a dire all'utente non solo quanto vale un biomarcatore, ma quanto
    quella patch sia tipica o anomala rispetto alla propria classe.
    """
    values = np.asarray(distribution, dtype=float)
    values = values[~np.isnan(values)]
    if values.size <= 1:
        return float("nan")
    return float((values < value).sum() / (values.size - 1) * 100.0)


# ---------------------------------------------------------------------------
# Elaborazione di un'immagine nuova
# ---------------------------------------------------------------------------
def load_reference_image(fase1_dir: Path, raw_dir: Path) -> np.ndarray:
    """
    Reference image di Macenko usata dalla Fase 1, in RGB.

    E' registrata in preprocessing_metadata.json: rileggerla da li' evita di
    riselezionarla scandendo tutte le 600 immagini a ogni avvio, e garantisce
    che la GUI normalizzi esattamente come la pipeline.
    """
    metadata_path = Path(fase1_dir) / "preprocessing_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} non trovato: eseguire prima la Fase 1 della pipeline."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    basename = metadata["reference_image"]["reference_image_basename"]

    for category in CATEGORIES:
        candidate = Path(raw_dir) / category / basename
        if candidate.exists():
            return cv2.cvtColor(_imread(candidate, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    raise FileNotFoundError(f"Reference image '{basename}' non trovata in {raw_dir}")


def build_normalizer(reference_rgb: np.ndarray):
    """Normalizzatore di Macenko fittato sulla reference image."""
    normalizer = _prep.StainNormalizerMacenko()
    normalizer.fit(reference_rgb)
    return normalizer


def process_image(image_rgb: np.ndarray, normalizer) -> dict:
    """
    Esegue Fase 1 -> 2 -> 3 su una singola immagine RGB.

    Ogni passo chiama la stessa funzione usata da run_pipeline.py: la GUI non
    reimplementa nulla, cosi' i valori mostrati coincidono con quelli dei CSV.

    Returns:
        dict con gli stadi intermedi ('normalized', 'denoised', 'h_channel',
        'mask', 'overlay'), la lista dei nuclei e le 47 feature per patch.
    """
    normalized = normalizer.transform(image_rgb)
    denoised = _prep.apply_bilateral_denoising(normalized)
    h_channel = _prep.extract_hematoxylin_channel_clahe(denoised)

    instance_mask, centroids = _seg.segment_nuclei_watershed(h_channel)
    instance_mask = instance_mask.astype(np.int32)

    nuclei = _feat.extract_nucleus_morphometry(instance_mask)
    features = _feat.aggregate_patch_morphometry(nuclei, "immagine_caricata", CATEGORIES[0])
    features.update(_feat.compute_knn_spatial_features(nuclei))
    features.update(_feat.extract_texture_features(h_channel, instance_mask))

    # image_name e category servono alla firma di aggregate_patch_morphometry ma
    # non sono biomarcatori: la GUI mostra solo le 47 feature.
    for metadata_column in ("image_name", "category"):
        features.pop(metadata_column, None)

    return {
        "normalized": normalized,
        "denoised": denoised,
        "h_channel": h_channel,
        "mask": instance_mask,
        "overlay": _seg.draw_segmentation_overlay(denoised, instance_mask, centroids),
        "nuclei": nuclei,
        "features": features,
    }


# ---------------------------------------------------------------------------
# Collegamento del classificatore della Fase 4
#
# Il modello non usa tutti e 47 i biomarcatori: la Fase 4 ne scarta 14 perche'
# ridondanti (|rho| > 0.90), e l'artefatto salvato porta con se' l'elenco dei 33
# sopravvissuti. Passare le colonne in un ordine diverso produrrebbe predizioni
# sbagliate senza che nulla protesti, quindi la selezione avviene sempre per
# nome, mai per posizione.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Classifier:
    """Modello della Fase 4 con i biomarcatori su cui e' stato addestrato."""

    model: object            # Pipeline scikit-learn
    features: list[str]


@dataclass(frozen=True)
class LocalExplanation:
    """Perche' il modello ha deciso cosi' per una singola patch."""

    contributions: object    # DataFrame: feature, value, contribution
    expected_value: float    # uscita media del modello sul dataset
    raw_output: float        # uscita per questa patch
    probability: float


def load_classifier(fase4_dir: Path) -> Classifier:
    """
    Carica il modello scelto dalla Fase 4.

    Raises:
        FileNotFoundError: se il modello non e' stato ancora prodotto.
    """
    import joblib

    model_path = Path(fase4_dir) / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} non trovato: eseguire prima `python src/04_classification.py`."
        )
    payload = joblib.load(model_path)
    return Classifier(model=payload["model"], features=list(payload["features"]))


def _feature_row(classifier: Classifier, features: dict) -> np.ndarray:
    """I biomarcatori richiesti dal modello, nell'ordine in cui li vuole."""
    missing = [name for name in classifier.features if name not in features]
    if missing:
        raise KeyError(
            f"biomarcatori mancanti per la predizione: {missing}. "
            "Sono quelli su cui il modello e' stato addestrato."
        )
    return np.array([[float(features[name]) for name in classifier.features]])


def predict_patch(classifier: Classifier, features: dict) -> float:
    """Probabilita' che la patch sia linfoma follicolare."""
    return float(classifier.model.predict_proba(_feature_row(classifier, features))[0, 1])


def explain_patch(classifier: Classifier, features: dict) -> LocalExplanation:
    """
    Contributo di ciascun biomarcatore alla decisione su QUESTA patch.

    Vale la proprieta' fondativa di SHAP: valore atteso + somma dei contributi =
    uscita del modello. E' cio' che rende il grafico a cascata una spiegazione e
    non un disegno, ed e' verificato da un test.

    Il modello scelto dalla Fase 4 e' basato su alberi (XGBoost): per questi
    l'explainer e' esatto, non approssimato.
    """
    import pandas as pd
    import shap

    row = _feature_row(classifier, features)
    steps = getattr(classifier.model, "steps", None)
    estimator = steps[-1][1] if steps else classifier.model
    transformed = classifier.model[:-1].transform(row) if steps and len(steps) > 1 else row

    explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3:              # (1, n_feature, 2) per la classificazione binaria
        values = values[:, :, 1]
    contributions = values[0]

    expected = float(np.atleast_1d(explainer.expected_value)[-1])

    # L'uscita va ricalcolata dal modello, non dedotta dai contributi: e' cio'
    # che rende verificabile l'additivita' invece di darla per scontata.
    if type(estimator).__name__.startswith("XGB"):
        raw_output = float(estimator.predict(transformed, output_margin=True)[0])
    else:
        raw_output = float(estimator.predict_proba(transformed)[0, 1])

    table = pd.DataFrame({
        "feature": classifier.features,
        "value": [float(features[name]) for name in classifier.features],
        "contribution": contributions,
    }).sort_values("contribution", key=np.abs, ascending=False, ignore_index=True)

    return LocalExplanation(
        contributions=table,
        expected_value=expected,
        raw_output=raw_output,
        probability=predict_patch(classifier, features),
    )


def load_out_of_fold_predictions(fase4_dir: Path):
    """
    Predizioni fuori-piega registrate dalla Fase 4.

    Per una patch del dataset e' l'unica predizione onesta: quella prodotta
    nella piega in cui la patch era in test, da un modello che non l'aveva mai
    vista in addestramento. Rieseguire il modello finale su una patch che ha gia'
    visto darebbe un numero ottimistico e privo di significato.
    """
    import pandas as pd

    path = Path(fase4_dir) / "out_of_fold_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non trovato: eseguire prima `python src/04_classification.py`."
        )
    return pd.read_csv(path)
