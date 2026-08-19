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
