"""Quanto il metodo di segmentazione sposta i biomarcatori.

La verifica umana della Fase 2 ha stabilito che il limite del Watershed non e' la
rilevazione ma la separazione: 83 fusioni assorbono il 10,3% dei nuclei marcati.
Una fusione non e' un nucleo perso, e' un oggetto con area doppia e forma
alterata, quindi entra nel calcolo delle grandezze morfometriche. Questo modulo
misura di quanto.

Il confronto avviene sulle dieci patch di validazione, le sole per cui esistono
sia le maschere del Watershed sia quelle di Cellpose. Dalle une e dalle altre si
estraggono i 47 biomarcatori con le stesse funzioni della Fase 3, e si misura lo
scarto fra le due matrici.

CHE COSA MISURA E CHE COSA NO. Misura lo spostamento dei biomarcatori dovuto al
diverso partizionamento in istanze. NON dice quale segmentazione classifichi
meglio: quella domanda richiederebbe di riaddestrare il modello sull'intero
dataset segmentato con Cellpose, quindi di rifare ogni numero delle fasi
successive.

Lo scarto e' espresso in frazioni dello scarto interquartile del dataset
completo, come nella prova di robustezza alla colorazione: uno spostamento di
1.0 significherebbe che il cambio di segmentazione muove il biomarcatore quanto
l'intera ampiezza interquartile fra le 600 patch.

Uso:  py src/segmentation_sensitivity.py
Esce: data/fase2_segmentation/segmentation_sensitivity.csv
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GT_DIR = BASE_DIR / "data" / "ground_truth" / "cellpose_v4"
FASE1_DIR = BASE_DIR / "data" / "fase1_preprocessing"
FASE2_DIR = BASE_DIR / "data" / "fase2_segmentation"
FASE3_CSV = BASE_DIR / "data" / "fase3_features" / "features_patches_master.csv"
OUTPUT_CSV = FASE2_DIR / "segmentation_sensitivity.csv"

CATEGORY_FL = "follicular_lymphoma"
CATEGORY_REACTIVE = "reactive_tissue"


def _load_extraction_module():
    """Il modulo della Fase 3 inizia con una cifra: import per percorso."""
    spec = importlib.util.spec_from_file_location(
        "mod_features", BASE_DIR / "src" / "03_feature_extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _category_of(stem: str) -> str:
    return CATEGORY_FL if stem.startswith("FL_") else CATEGORY_REACTIVE


def patch_features(mask: np.ndarray, h_channel: np.ndarray, stem: str, extraction) -> dict:
    """I 47 biomarcatori di una patch, nella stessa sequenza della Fase 3."""
    nuclei = extraction.extract_nucleus_morphometry(mask)
    features = extraction.aggregate_patch_morphometry(nuclei, stem, _category_of(stem))
    features.update(extraction.compute_knn_spatial_features(nuclei))
    features.update(extraction.extract_texture_features(h_channel, mask))
    return features


def paired_matrices(extraction) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Biomarcatori delle stesse patch, da Watershed e da Cellpose."""
    righe_ws, righe_cp = [], []

    for gt_path in sorted(GT_DIR.glob("*_cellpose_gt.png")):
        stem = gt_path.name.replace("_cellpose_gt.png", "")
        category = _category_of(stem)

        mask_cp = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        mask_ws = cv2.imread(
            str(FASE2_DIR / category / "masks" / f"{stem}_mask.png"), cv2.IMREAD_UNCHANGED
        )
        h_channel = cv2.imread(
            str(FASE1_DIR / category / "h_channel" / f"{stem}_hchannel.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask_cp is None or mask_ws is None or h_channel is None:
            raise FileNotFoundError(f"input mancante per {stem}")

        righe_ws.append(patch_features(mask_ws, h_channel, stem, extraction))
        righe_cp.append(patch_features(mask_cp, h_channel, stem, extraction))

    return pd.DataFrame(righe_ws), pd.DataFrame(righe_cp)


def shift_table(watershed: pd.DataFrame, cellpose: pd.DataFrame,
                dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Scarto fra le due segmentazioni, per biomarcatore.

    Lo scarto mediano e' normalizzato sullo scarto interquartile del dataset
    completo: senza normalizzazione un'area in micrometri quadrati e una
    frazione adimensionale non sarebbero confrontabili.
    """
    metadata = set(getattr(_EXTRACTION, "PATCH_METADATA_COLUMNS", ()))
    colonne = [c for c in watershed.columns
               if c not in metadata and pd.api.types.is_numeric_dtype(watershed[c])]

    righe = []
    for colonna in colonne:
        differenze = cellpose[colonna].to_numpy(float) - watershed[colonna].to_numpy(float)
        iqr = float(dataset[colonna].quantile(0.75) - dataset[colonna].quantile(0.25))
        righe.append({
            "feature": colonna,
            "mediana_watershed": float(np.median(watershed[colonna])),
            "mediana_cellpose": float(np.median(cellpose[colonna])),
            "scarto_mediano": float(np.median(differenze)),
            "scarto_in_iqr": float(np.median(np.abs(differenze)) / iqr) if iqr > 0 else np.nan,
            "patch_con_aumento": int((differenze > 0).sum()),
        })

    return (pd.DataFrame(righe)
            .sort_values("scarto_in_iqr", ascending=False)
            .reset_index(drop=True))


_EXTRACTION = _load_extraction_module()


if __name__ == "__main__":
    watershed, cellpose = paired_matrices(_EXTRACTION)
    dataset = pd.read_csv(FASE3_CSV)

    tabella = shift_table(watershed, cellpose, dataset)
    tabella.to_csv(OUTPUT_CSV, index=False)

    print(f"[Segmentazione] Patch confrontate: {len(watershed)}")
    print(f"[Segmentazione] Nuclei: Watershed {int(watershed['n_nuclei'].sum())}, "
          f"Cellpose {int(cellpose['n_nuclei'].sum())}")
    print(f"[Segmentazione] Scarto mediano in unita' di IQR: "
          f"{tabella['scarto_in_iqr'].median():.3f}")
    print(tabella.head(10).to_string(index=False))
    print(f"[Segmentazione] -> {OUTPUT_CSV}")
