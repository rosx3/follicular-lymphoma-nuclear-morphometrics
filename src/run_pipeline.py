"""
===============================================================================
run_pipeline.py — Script di Orchestrazione della Pipeline Completa
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
      tramite Morfometria Nucleare e AI Interpretabile (White-Box XAI)
===============================================================================
Questo script è il punto di ingresso principale (entry point) per eseguire
l'intera pipeline di analisi dall'inizio alla fine.

Fasi della Pipeline:
  Fase 1 — Preprocessing:    Macenko Stain Normalization → Filtro Bilaterale → CLAHE
  Fase 2 — Segmentazione:    Watershed Distance Transform → Estrazione Centroidi
  Fase 3 — Biomarcatori:     Citomorfometria + Grafi Spaziali + Tessitura GLCM/LBP  [TODO]
  Fase 4 — Classificazione:  Random Forest / XGBoost + SHAP XAI                     [TODO]

Utilizzo:
  python src/run_pipeline.py                    # Esegue Fase 1 + Fase 2
  python src/run_pipeline.py --fase 1           # Solo Fase 1 (Preprocessing)
  python src/run_pipeline.py --fase 2           # Solo Fase 2 (Segmentazione)
  python src/run_pipeline.py --fase 1 2         # Fase 1 e Fase 2 (default)

Struttura attesa del dataset (data/raw/):
  data/raw/
  ├── follicular_lymphoma/     # 300 immagini JPEG 224x224 px
  │   ├── FL_examples (1).jpg
  │   └── ...
  └── reactive_tissue/         # 300 immagini JPEG 224x224 px
      ├── REACTIVE_examples (1).jpg
      └── ...

Output generati:
  data/fase1_preprocessing/    # H-channel + RGB normalizzate (600 immagini)
  data/fase2_segmentation/     # Maschere 16-bit + Overlay + centroids_all.csv
===============================================================================
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import importlib.util

_SRC_DIR = Path(__file__).resolve().parent

# src/ non e' un package: va reso importabile per i moduli con nome regolare.
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Convenzioni di naming e categorie — unica fonte di verita' (vedi src/naming.py)
from naming import (  # noqa: E402
    CATEGORIES,
    CATEGORY_FL,
    PatchInputs,
    h_channel_name,
    iter_h_channel_inputs,
    iter_patch_inputs,
    mask_name,
    overlay_name,
    rgb_normalized_name,
    short_label,
    target_from_category,
)

# Modulo 01: Preprocessing
_spec_prep = importlib.util.spec_from_file_location("mod_preprocessing", _SRC_DIR / "01_preprocessing.py")
_mod_prep = importlib.util.module_from_spec(_spec_prep)
_spec_prep.loader.exec_module(_mod_prep)

find_best_reference_image = _mod_prep.find_best_reference_image
StainNormalizerMacenko    = _mod_prep.StainNormalizerMacenko
process_single_image      = _mod_prep.process_single_image

# Modulo 02: Segmentazione
_spec_seg = importlib.util.spec_from_file_location("mod_segmentation", _SRC_DIR / "02_segmentation.py")
_mod_seg = importlib.util.module_from_spec(_spec_seg)
_spec_seg.loader.exec_module(_mod_seg)

segment_nuclei_watershed  = _mod_seg.segment_nuclei_watershed
draw_segmentation_overlay = _mod_seg.draw_segmentation_overlay

# Modulo 03: Feature Extraction
_spec_feat = importlib.util.spec_from_file_location("mod_feature_extraction", _SRC_DIR / "03_feature_extraction.py")
_mod_feat = importlib.util.module_from_spec(_spec_feat)
_spec_feat.loader.exec_module(_mod_feat)

extract_nucleus_morphometry    = _mod_feat.extract_nucleus_morphometry
aggregate_patch_morphometry    = _mod_feat.aggregate_patch_morphometry
save_morphometry_visual_preview = _mod_feat.save_morphometry_visual_preview



# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
RAW_DIR       = BASE_DIR / "data" / "raw"
FASE1_DIR     = BASE_DIR / "data" / "fase1_preprocessing"
FASE2_DIR     = BASE_DIR / "data" / "fase2_segmentation"
FASE3_DIR     = BASE_DIR / "data" / "fase3_features"
IMG_FASE3_DIR = BASE_DIR / "img" / "fase3"


# ---------------------------------------------------------------------------
# Fase 1 — Preprocessing
# ---------------------------------------------------------------------------
def run_fase1(verbose: bool = True) -> None:
    """
    Esegue la pipeline di preprocessing su tutte le immagini in data/raw/.

    Per ogni immagine:
      1. Normalizzazione cromatica di Macenko (Reference Image automatica)
      2. Denoising con Filtro Bilaterale
      3. Estrazione Canale Ematossilina (H-channel) + CLAHE

    Output:
      data/fase1_preprocessing/<categoria>/h_channel/       ← canale H CLAHE
      data/fase1_preprocessing/<categoria>/rgb_normalized/  ← RGB normalizzata
    """
    print("\n" + "=" * 70)
    print("  FASE 1: Preprocessing Istologico (Macenko + Bilaterale + CLAHE)")
    print("=" * 70)

    # Individua le directory raw disponibili
    raw_dirs = [RAW_DIR / cat for cat in CATEGORIES]
    raw_dirs = [d for d in raw_dirs if d.exists()]

    if not raw_dirs:
        raise FileNotFoundError(f"Nessuna cartella trovata in {RAW_DIR}. "
                                "Assicurarsi che il dataset sia in data/raw/.")

    # Selezione automatica della Reference Image tra tutte le immagini
    if verbose:
        print("[Fase 1] Selezione automatica Reference Image...")
    ref_path, _ = find_best_reference_image([str(d) for d in raw_dirs])
    if verbose:
        print(f"[Fase 1] Reference Image: {ref_path}")

    # Inizializza il normalizzatore di Macenko.
    # find_best_reference_image restituisce un path, mentre fit() — come
    # transform() e process_single_image() — lavora su array RGB: la reference
    # va caricata e convertita da BGR prima del fit.
    ref_bgr = cv2.imread(str(ref_path))
    if ref_bgr is None:
        raise ValueError(f"Impossibile caricare la Reference Image: {ref_path}")

    normalizer = StainNormalizerMacenko()
    normalizer.fit(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB))

    # Contatori
    n_ok = 0
    n_err = 0
    t0 = time.time()

    for cat_dir in raw_dirs:
        cat_name = cat_dir.name  # es. "follicular_lymphoma"
        img_paths = sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.png"))

        # Prepara le directory di output
        out_h   = FASE1_DIR / cat_name / "h_channel"
        out_rgb = FASE1_DIR / cat_name / "rgb_normalized"
        out_h.mkdir(parents=True, exist_ok=True)
        out_rgb.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(img_paths, desc=f"  Fase 1 — {cat_name}", unit="img"):
            try:
                result = process_single_image(str(img_path), normalizer=normalizer)

                # Salva H-channel (uint8 grayscale)
                cv2.imwrite(
                    str(out_h / h_channel_name(img_path.stem)),
                    result["hematoxylin_h"]
                )
                # Salva RGB normalizzata (uint8 BGR per OpenCV)
                rgb_bgr = cv2.cvtColor(result["denoised_rgb"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(out_rgb / rgb_normalized_name(img_path.stem)),
                    rgb_bgr
                )
                n_ok += 1

            except Exception as e:
                print(f"\n  [WARN] Errore su {img_path.name}: {e}")
                n_err += 1

    elapsed = time.time() - t0
    print(f"\n[Fase 1] Completata: {n_ok} immagini OK, {n_err} errori — "
          f"{elapsed:.1f}s ({elapsed/max(n_ok,1):.2f}s/img)")


# ---------------------------------------------------------------------------
# Fase 2 — Segmentazione
# ---------------------------------------------------------------------------
def run_fase2(verbose: bool = True) -> None:
    """
    Esegue la segmentazione d'istanza su tutti gli H-channel generati dalla Fase 1.

    Per ogni H-channel:
      1. Marker-Controlled Distance Transform Watershed
      2. Estrazione Centroidi (coordinate px e µm)
      3. Generazione Maschera 16-bit PNG e Overlay RGB

    Output:
      data/fase2_segmentation/<categoria>/masks/          ← maschere istanza 16-bit
      data/fase2_segmentation/<categoria>/overlays/       ← overlay RGB con contorni
      data/fase2_segmentation/centroids_all.csv           ← tutti i centroidi
    """
    print("\n" + "=" * 70)
    print("  FASE 2: Segmentazione Nuclei (Watershed v3.0) + Centroidi")
    print("=" * 70)

    all_centroids = []
    n_ok = 0
    n_err = 0
    t0 = time.time()

    for cat_name in CATEGORIES:
        out_masks    = FASE2_DIR / cat_name / "masks"
        out_overlays = FASE2_DIR / cat_name / "overlays"
        out_masks.mkdir(parents=True, exist_ok=True)
        out_overlays.mkdir(parents=True, exist_ok=True)

        # iter_h_channel_inputs risolve H-channel + RGB normalizzata e solleva
        # FileNotFoundError se un input manca: nessun overlay viene piu' saltato
        # in silenzio.
        images = list(iter_h_channel_inputs(cat_name, FASE1_DIR))

        for image in tqdm(images, desc=f"  Fase 2 — {cat_name}", unit="img"):
            try:
                stem = image.stem

                h_channel = cv2.imread(str(image.h_channel_path), cv2.IMREAD_GRAYSCALE)
                if h_channel is None:
                    raise ValueError(f"Impossibile leggere {image.h_channel_path.name}")

                # Segmentazione Watershed
                instance_mask, centroids = segment_nuclei_watershed(h_channel)

                # Salva maschera 16-bit
                mask_path = out_masks / mask_name(stem)
                cv2.imwrite(str(mask_path), instance_mask.astype(np.uint16))

                # Overlay visivo sulla RGB normalizzata corrispondente
                rgb_img = cv2.imread(str(image.rgb_path), cv2.IMREAD_COLOR)
                if rgb_img is None:
                    raise ValueError(f"Impossibile leggere {image.rgb_path.name}")
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
                overlay = draw_segmentation_overlay(rgb_img, instance_mask, centroids)
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(out_overlays / overlay_name(stem)),
                    overlay_bgr
                )

                # Aggrega centroidi al CSV master
                for c in centroids:
                    all_centroids.append({
                        "image_name":       stem,
                        "category":         cat_name,
                        "nucleus_id":       c["id"],
                        "centroid_x_px":    c["centroid_x_px"],
                        "centroid_y_px":    c["centroid_y_px"],
                        "centroid_x_um":    round(c["centroid_x_px"] * 0.23, 3),
                        "centroid_y_um":    round(c["centroid_y_px"] * 0.23, 3),
                        "area_px":          c.get("area_px", 0),
                        "area_um2":         round(c.get("area_px", 0) * (0.23 ** 2), 3),
                    })

                n_ok += 1

            except Exception as e:
                print(f"\n  [WARN] Errore su {image.h_channel_path.name}: {e}")
                n_err += 1

    # Salva CSV master dei centroidi
    if all_centroids:
        csv_path = FASE2_DIR / "centroids_all.csv"
        FASE2_DIR.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_centroids[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_centroids)
        print(f"\n[Fase 2] CSV Master: {len(all_centroids):,} nuclei → {csv_path}")

    elapsed = time.time() - t0
    print(f"[Fase 2] Completata: {n_ok} immagini OK, {n_err} errori — "
          f"{elapsed:.1f}s ({elapsed/max(n_ok,1):.2f}s/img)")


# ---------------------------------------------------------------------------
# Fase 3 — Estrazione Biomarcatori Citomorfometrici
# ---------------------------------------------------------------------------
def run_fase3(verbose: bool = True) -> None:
    """
    Esegue l'estrazione dei biomarcatori citomorfometrici d'istanza e aggregati.

    Input:
      data/fase2_segmentation/<categoria>/masks/   ← maschere 16-bit
      data/fase1_preprocessing/<categoria>/rgb_normalized/ ← RGB per anteprime

    Output:
      data/fase3_features/features_nuclei_all.csv   ← ~94.042 righe (nuclei singoli)
      data/fase3_features/features_patches_master.csv ← 600 righe (patch aggregate)
      data/fase3_features/feature_extraction_metadata.json ← Metadati FAIR
      img/fase3/morphometry_regions_preview.png     ← Anteprima Bounding Boxes
    """
    print("\n" + "=" * 70)
    print("  FASE 3: Estrazione Biomarcatori Citomorfometrici (Forma & Dimensione)")
    print("=" * 70)

    FASE3_DIR.mkdir(parents=True, exist_ok=True)
    IMG_FASE3_DIR.mkdir(parents=True, exist_ok=True)

    all_nuclei_rows = []
    all_patch_rows  = []
    n_ok = 0
    n_err = 0
    t0 = time.time()

    sample_fl: PatchInputs | None = None
    sample_re: PatchInputs | None = None

    for category in CATEGORIES:
        # iter_patch_inputs risolve maschera + RGB + H-channel secondo le
        # convenzioni di src/naming.py e solleva FileNotFoundError se un input
        # manca: nessuna patch viene piu' saltata in silenzio (audit B1).
        patches = list(iter_patch_inputs(category, FASE1_DIR, FASE2_DIR))

        for patch in tqdm(patches, desc=f"  Fase 3 — {short_label(category)}", unit="img"):
            try:
                mask_16 = cv2.imread(str(patch.mask_path), cv2.IMREAD_UNCHANGED)
                if mask_16 is None:
                    raise ValueError(f"Impossibile leggere {patch.mask_path.name}")

                # 1. Estrazione d'istanza per singolo nucleo
                nuclei_feat = extract_nucleus_morphometry(mask_16)

                # Salva campioni per l'anteprima grafica
                if category == CATEGORY_FL and sample_fl is None:
                    sample_fl = patch
                elif category != CATEGORY_FL and sample_re is None:
                    sample_re = patch

                for nf in nuclei_feat:
                    nf["image_name"] = patch.stem
                    nf["category"]   = category
                    all_nuclei_rows.append(nf)

                # 2. Aggregazione statistica per patch
                patch_stat = aggregate_patch_morphometry(
                    nuclei_feat, patch.stem, category
                )
                patch_stat["target"] = target_from_category(category)
                all_patch_rows.append(patch_stat)

                n_ok += 1

            except Exception as e:
                print(f"\n  [WARN] Errore su {patch.mask_path.name}: {e}")
                n_err += 1

    # 3. Salva CSV Singoli Nuclei (~94.042 righe)
    if all_nuclei_rows:
        csv_nuclei = FASE3_DIR / "features_nuclei_all.csv"
        # Riorganizza colonne con image_name e category all'inizio
        first_cols = ["image_name", "category", "nucleus_id"]
        other_cols = [c for c in all_nuclei_rows[0].keys() if c not in first_cols]
        fieldnames = first_cols + other_cols

        with open(csv_nuclei, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_nuclei_rows)
        print(f"\n[Fase 3] CSV Nuclei Singoli: {len(all_nuclei_rows):,} nuclei → {csv_nuclei}")

    # 4. Salva CSV Patch Master (600 righe)
    if all_patch_rows:
        csv_patches = FASE3_DIR / "features_patches_master.csv"
        fieldnames = list(all_patch_rows[0].keys())
        with open(csv_patches, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_patch_rows)
        print(f"[Fase 3] CSV Patch Master: {len(all_patch_rows)} patch → {csv_patches}")

    # 5. Generazione Anteprima Grafica con Bounding Boxes
    if sample_fl is None or sample_re is None:
        raise RuntimeError(
            "Anteprima citomorfometrica non generabile: manca almeno una patch "
            f"campione (FL={sample_fl is not None}, REACTIVE={sample_re is not None})."
        )

    preview_png = IMG_FASE3_DIR / "morphometry_regions_preview.png"
    save_morphometry_visual_preview(
        str(sample_fl.rgb_path), str(sample_fl.mask_path),
        str(sample_re.rgb_path), str(sample_re.mask_path),
        str(preview_png)
    )

    elapsed = time.time() - t0
    print(f"[Fase 3] Completata: {n_ok} patch OK, {n_err} errori — "
          f"{elapsed:.1f}s ({elapsed/max(n_ok,1):.2f}s/img)")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Morfometria Nucleare — Linfoma Follicolare vs Tessuto Reattivo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--fase",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 3],
        metavar="N",
        help=(
            "Fasi da eseguire (default: 1 2 3).\n"
            "  1 = Preprocessing (Macenko + CLAHE)\n"
            "  2 = Segmentazione (Watershed + Centroidi)\n"
            "  3 = Estrazione Biomarcatori Citomorfometrici\n"
            "  4 = Classificazione + SHAP XAI [TODO]"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Modalità silenziosa")
    args = parser.parse_args()

    verbose = not args.quiet
    fasi = sorted(set(args.fase))

    print("\n" + "=" * 70)
    print("  Pipeline: Morfometria Nucleare per Diagnosi Linfoma Follicolare")
    print(f"  Fasi selezionate: {fasi}")
    print("=" * 70)

    if 1 in fasi:
        run_fase1(verbose=verbose)

    if 2 in fasi:
        run_fase2(verbose=verbose)

    if 3 in fasi:
        run_fase3(verbose=verbose)

    if 4 in fasi:
        print("\n[INFO] Fase 4 (Classificazione + XAI) — in sviluppo.")
        # TODO: importare e invocare src/04_classification.py

    print("\n✅ Pipeline completata.")


if __name__ == "__main__":
    main()
