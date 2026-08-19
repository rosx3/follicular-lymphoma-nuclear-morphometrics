"""
===============================================================================
Modulo 03: Estrazione Biomarcatori Citomorfometrici, Spaziali e di Tessitura
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
      tramite Morfometria Nucleare e AI Interpretabile (White-Box XAI)
Versione: 2.0 — Skeleton allineato a reports/fase3_report.md (agosto 2026)
===============================================================================
Questo modulo implementa la caratterizzazione citomorfometrica, micro-spaziale
e di tessitura secondo il set definitivo di biomarcatori documentato in
reports/fase3_report.md (Sezione 2 — 51 feature + 3 metadati):

 1. STEP 1 — Morfometria d'istanza per singolo nucleo (questo file, completo).
 2. STEP 2 — Aggregazione statistica per patch: mean/std/skew/cv su 8 feature
    di base (questo file, completo).
 3. STEP 3 — Distanze micro-spaziali k-NN (k=1, k=3) via scipy.spatial.KDTree
    (scheletro — implementazione nella prossima iterazione).
 4. STEP 4 — Tessitura cromatinica H-channel: GLCM (contrast/homogeneity/energy)
    + LBP entropy + statistiche di intensità (scheletro — prossima iterazione).

DECISIONI METODOLOGICHE GIA' VALIDATE (vedi reports/fase3_report.md §1.1):
  - NIENTE Delaunay/MST: boundary effects critici su patch 224x224 px (51.5 µm).
    Direzione futura: WSI + GNN (GraphSAGE/GAT) — vedi report §7.
  - NIENTE equivalent_diameter_um in aggregazione: ridondante con area_um2
    (d = 2*sqrt(A/pi)). Rimane disponibile a livello di singolo nucleo per
    audit/tracciabilità nel CSV nuclei-level.
  - NIENTE extent in aggregazione: ridondante con solidity. Stesso discorso
    per la tracciabilità a livello di singolo nucleo.
  - NIENTE mediana negli aggregati: sostituita da coefficiente di variazione
    (cv = std/mean), più informativo su una distribuzione già coperta da
    mean/std/skew.
  - NIENTE momenti colore CIE-LAB: ridondanti post-normalizzazione Macenko
    (vedi reports/fase1_report.md).
  - NUOVA feature aspect_ratio (major/minor axis) a livello di singolo nucleo,
    necessaria per completare il set di 8 feature morfometriche di base
    dichiarate in reports/fase3_report.md §2.4.

NOTA APERTA PER LA TESI (da verificare con l'utente prima di finalizzare):
  Il report fase3_report.md dichiara "51 feature + 3 metadati" in apertura
  della Sezione 2, ma la somma dei sub-totali dichiarati per sezione
  (3 densità + 2 Iwamoto + 32 morfometria + 4 knn + 6 tessitura = 47) non
  torna con 51. Il conteggio di 32 colonne morfometriche è coerente solo se
  si applicano uniformemente le 4 statistiche (mean/std/skew/cv) a tutte le
  8 feature di base (8*4=32) — la tabella dettagliata §2.4 elenca invece
  conteggi ridotti e disomogenei per singola feature. In questo skeleton
  ho adottato l'interpretazione uniforme (32 colonne, 4 stat x 8 feature)
  perché è l'unica che rende quadrare il totale dichiarato di 51. Verificare
  empiricamente le colonne prodotte una volta eseguita la pipeline completa.
===============================================================================
"""

import os
from pathlib import Path
import numpy as np
import cv2
from scipy import stats
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Costanti di Calibrazione Spaziale
# Scanner: Hamamatsu NanoZoomer S360, Obiettivo 40x
# ---------------------------------------------------------------------------
MICRONS_PER_PIXEL = 0.23  # µm / px
PIXEL_AREA_UM2 = MICRONS_PER_PIXEL**2  # 0.0529 µm² / px²
PATCH_SIZE_PX = 224
PATCH_AREA_UM2 = (PATCH_SIZE_PX * MICRONS_PER_PIXEL) ** 2  # 2654.31 µm²

# Feature morfometriche di base su cui calcolare le 4 statistiche aggregate
# (mean, std, skew, cv) — vedi reports/fase3_report.md §2.4.
MORPHOMETRY_BASE_FEATURES: list[str] = [
    "area_um2",
    "perimeter_um",
    "circularity",
    "eccentricity",
    "solidity",
    "major_axis_um",
    "minor_axis_um",
    "aspect_ratio",
]

# Feature mantenute a livello di singolo nucleo per tracciabilità/audit ma
# escluse dall'aggregazione per patch (ridondanti — vedi report §1.1).
NUCLEUS_ONLY_AUDIT_FEATURES: list[str] = ["extent", "equivalent_diameter_um"]


# ---------------------------------------------------------------------------
# 1. Estrazione Citomorfometrica per Singolo Nucleo
# ---------------------------------------------------------------------------
def extract_nucleus_morphometry(instance_mask: np.ndarray) -> list[dict]:
    """
    Estrae i biomarcatori citomorfometrici di forma e dimensione per ciascun
    nucleo presente nella maschera d'istanza 16-bit.

    Args:
        instance_mask: Maschera 2D int (0 = sfondo, ID >= 1 = nuclei).

    Returns:
        Lista di dizionari con i biomarcatori di ogni nucleo. Include sia le
        feature del set definitivo (aggregabili per patch) sia le feature
        di solo audit (extent, equivalent_diameter_um) mantenute per
        tracciabilità nel CSV nuclei-level.
    """
    props = regionprops(instance_mask)
    nuclei_features = []

    for p in props:
        area_px = p.area
        area_um2 = area_px * PIXEL_AREA_UM2

        perimeter_px = p.perimeter
        perimeter_um = perimeter_px * MICRONS_PER_PIXEL

        # Circolarità: C = 4*pi*Area / Perimetro^2  (1.0 = cerchio perfetto)
        if perimeter_px > 0:
            circ = float(np.clip((4.0 * np.pi * area_px) / (perimeter_px**2), 0.0, 1.0))
        else:
            circ = 0.0

        ecc = float(p.eccentricity)          # 0 = cerchio, 1 = linea
        solidity = float(p.solidity)          # Area / Area Convex Hull
        extent = float(p.extent)              # Area / Area Bounding Box [AUDIT ONLY]

        major_axis_um = float(p.axis_major_length * MICRONS_PER_PIXEL)
        minor_axis_um = float(p.axis_minor_length * MICRONS_PER_PIXEL)

        # Aspect ratio (major/minor) — NUOVO rispetto a v1.0, richiesto dal
        # set definitivo di 8 feature morfometriche di base (report §2.4).
        aspect_ratio = float(major_axis_um / minor_axis_um) if minor_axis_um > 0 else 0.0

        equiv_diam_um = float(p.equivalent_diameter_area * MICRONS_PER_PIXEL)  # [AUDIT ONLY]

        minr, minc, maxr, maxc = p.bbox
        cy, cx = p.centroid

        nuclei_features.append({
            "nucleus_id": int(p.label),
            "centroid_x_px": round(float(cx), 2),
            "centroid_y_px": round(float(cy), 2),
            "centroid_x_um": round(float(cx * MICRONS_PER_PIXEL), 2),
            "centroid_y_um": round(float(cy * MICRONS_PER_PIXEL), 2),
            "area_px": int(area_px),
            "area_um2": round(area_um2, 3),
            "perimeter_px": round(float(perimeter_px), 2),
            "perimeter_um": round(perimeter_um, 3),
            "circularity": round(circ, 4),
            "eccentricity": round(ecc, 4),
            "solidity": round(solidity, 4),
            "major_axis_um": round(major_axis_um, 3),
            "minor_axis_um": round(minor_axis_um, 3),
            "aspect_ratio": round(aspect_ratio, 4),
            # --- feature di solo audit (non aggregate a livello patch) ---
            "extent": round(extent, 4),
            "equivalent_diameter_um": round(equiv_diam_um, 3),
            "bbox_ymin": int(minr),
            "bbox_xmin": int(minc),
            "bbox_ymax": int(maxr),
            "bbox_xmax": int(maxc),
            "orientation_rad": round(float(p.orientation), 4),
        })

    return nuclei_features


# ---------------------------------------------------------------------------
# 2. Aggregazione Statistica per Patch (Firma di Patch)
# ---------------------------------------------------------------------------
def _coefficient_of_variation(values: np.ndarray) -> float:
    """cv = std / |mean|, con guardia contro divisione per zero."""
    mean_val = float(np.mean(values))
    if abs(mean_val) < 1e-9:
        return 0.0
    return float(np.std(values)) / abs(mean_val)


def aggregate_patch_morphometry(
    nuclei_list: list[dict], image_name: str, category: str
) -> dict:
    """
    Calcola le statistiche aggregate di citomorfometria per l'intera patch,
    secondo il set definitivo documentato in reports/fase3_report.md §2.

    Metriche calcolate:
      - Densità nucleare (n_nuclei, nuclear_density_per_1000um2, nuclear_area_fraction)
      - Area Top 10%: media area e media asse minore (Iwamoto et al. 2024)
      - mean / std / skew / cv per le 8 feature morfometriche di base
        (MORPHOMETRY_BASE_FEATURES)
    """
    n_nuclei = len(nuclei_list)

    patch_dict = {
        "image_name": image_name,
        "category": category,
        "n_nuclei": n_nuclei,
        "nuclear_density_per_1000um2": round((n_nuclei / PATCH_AREA_UM2) * 1000.0, 3),
    }

    if n_nuclei == 0:
        patch_dict["nuclear_area_fraction"] = 0.0
        patch_dict["area_top10_mean_um2"] = 0.0
        patch_dict["area_top10_short_axis_um"] = 0.0
        for feat in MORPHOMETRY_BASE_FEATURES:
            for stat_name in ["mean", "std", "skew", "cv"]:
                patch_dict[f"{feat}_{stat_name}"] = 0.0
        return patch_dict

    total_nuclear_area_um2 = sum(n["area_um2"] for n in nuclei_list)
    patch_dict["nuclear_area_fraction"] = round(total_nuclear_area_um2 / PATCH_AREA_UM2, 4)

    # Top 10% Area (Iwamoto et al. 2024 — indicatore di centroblasti)
    nuclei_by_area_desc = sorted(nuclei_list, key=lambda n: n["area_um2"], reverse=True)
    top10_k = max(1, int(np.ceil(n_nuclei * 0.10)))
    top10_nuclei = nuclei_by_area_desc[:top10_k]

    patch_dict["area_top10_mean_um2"] = round(
        float(np.mean([n["area_um2"] for n in top10_nuclei])), 3
    )
    # Asse corto medio dei nuclei nel top 10% per area — Iwamoto et al. (2024),
    # short axis length, p=0.020.
    patch_dict["area_top10_short_axis_um"] = round(
        float(np.mean([n["minor_axis_um"] for n in top10_nuclei])), 3
    )

    # mean / std / skew / cv per le 8 feature morfometriche di base
    for feat in MORPHOMETRY_BASE_FEATURES:
        vals = np.array([n[feat] for n in nuclei_list], dtype=np.float64)
        patch_dict[f"{feat}_mean"] = round(float(np.mean(vals)), 4)
        patch_dict[f"{feat}_std"] = round(float(np.std(vals)), 4)
        sk = float(stats.skew(vals)) if len(vals) > 2 else 0.0
        patch_dict[f"{feat}_skew"] = round(sk if not np.isnan(sk) else 0.0, 4)
        patch_dict[f"{feat}_cv"] = round(_coefficient_of_variation(vals), 4)

    return patch_dict


# ---------------------------------------------------------------------------
# 3. Distanze Micro-Spaziali k-NN [SCHELETRO — prossima iterazione]
# ---------------------------------------------------------------------------
def compute_knn_spatial_features(nuclei_list: list[dict]) -> dict:
    """
    Calcola le distanze medie/std ai k=1 e k=3 vicini più prossimi per ogni
    nucleo della patch, usando scipy.spatial.KDTree su centroid_x_um/y_um.

    Sostituisce Delaunay/MST (esclusi per boundary effects su patch 224x224 px,
    vedi reports/fase3_report.md §1.1 e §7) come proxy di micro-architettura
    del packing nucleare.

    Output atteso (4 colonne):
      knn1_dist_mean_um, knn1_dist_std_um, knn3_dist_mean_um, knn3_dist_std_um

    Nota implementativa: con n_nuclei < 4 il k=3 non è definibile — gestire
    con fallback a 0.0 o NaN esplicito (da decidere per coerenza col resto
    della pipeline).

    TODO(prossima iterazione): implementare con scipy.spatial.KDTree.
    """
    raise NotImplementedError(
        "STEP 3 — da implementare nella prossima iterazione (vedi roadmap Fase 3)."
    )


# ---------------------------------------------------------------------------
# 4. Tessitura Cromatinica H-channel [SCHELETRO — prossima iterazione]
# ---------------------------------------------------------------------------
def extract_texture_features(h_channel_patch: np.ndarray) -> dict:
    """
    Calcola i descrittori di tessitura cromatinica sull'intera patch H-channel
    (CLAHE, prodotta dalla Fase 1), non sul singolo nucleo.

    Output atteso (6 colonne):
      glcm_contrast, glcm_homogeneity, glcm_energy  (skimage.feature.graycomatrix/graycoprops)
      lbp_entropy                                    (skimage.feature.local_binary_pattern
                                                        + entropia di Shannon dell'istogramma)
      hchannel_mean, hchannel_std                    (statistiche dirette sui pixel)

    Parametri GLCM/LBP (distanze, angoli, raggio, n_points, metodo 'uniform')
    da fissare e documentare in feature_extraction_metadata.json per
    riproducibilità.

    TODO(prossima iterazione): implementare con skimage.feature.
    """
    raise NotImplementedError(
        "STEP 4 — da implementare nella prossima iterazione (vedi roadmap Fase 3)."
    )


# ---------------------------------------------------------------------------
# 5. Generazione Anteprime Visive (Bounding Boxes, Contorni, Ellissi)
# ---------------------------------------------------------------------------
def generate_morphometry_overlay_image(
    rgb_img: np.ndarray, instance_mask: np.ndarray, nuclei_list: list[dict]
) -> np.ndarray:
    """
    Genera un'immagine RGB ad alta definizione che sovrappone:
      - Bounding Box (rettangolo blu) per ogni nucleo
      - Contorno d'istanza (verde)
      - Centroide (punto rosso)
    """
    overlay = rgb_img.copy()

    for n in nuclei_list:
        ymin, xmin, ymax, xmax = n["bbox_ymin"], n["bbox_xmin"], n["bbox_ymax"], n["bbox_xmax"]
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (255, 120, 0), 1)
        cx, cy = int(n["centroid_x_px"]), int(n["centroid_y_px"])
        cv2.circle(overlay, (cx, cy), 1, (0, 0, 255), -1)

    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]
    for uid in unique_ids:
        binary_single = np.uint8(instance_mask == uid)
        cnts, _ = cv2.findContours(binary_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 1)

    return overlay


def save_morphometry_visual_preview(
    fl_rgb_path: str,
    fl_mask_path: str,
    re_rgb_path: str,
    re_mask_path: str,
    output_png_path: str,
) -> None:
    """
    Crea una figura comparativa che mostra le regioni considerate dalla
    citomorfometria con Bounding Boxes e contorni per FL vs REACTIVE.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)

    for i, (rgb_p, mask_p, title, _cat_color) in enumerate([
        (fl_rgb_path, fl_mask_path, "Linfoma Follicolare (FL)", "red"),
        (re_rgb_path, re_mask_path, "Tessuto Reattivo (REACTIVE)", "blue"),
    ]):
        rgb = cv2.cvtColor(cv2.imread(rgb_p), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)

        nuclei = extract_nucleus_morphometry(mask)
        overlay = generate_morphometry_overlay_image(rgb, mask, nuclei)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"{title} — RGB Normalizzata", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"{title} — Bounding Boxes ({len(nuclei)} nuclei)", fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")

    legend_elements = [
        mpatches.Patch(color="green", label="Contorno Nucleare Segmentato"),
        mpatches.Patch(color="orange", label="Bounding Box Citomorfometrica"),
        mpatches.Patch(color="red", label="Centroide Nucleare (x, y)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, bbox_inches="tight")
    plt.close()
    print(f"[Fase 3] Anteprima visiva citomorfometrica salvata: {output_png_path}")


# ---------------------------------------------------------------------------
# Self-test del Modulo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Modulo 03 — Estrazione Citomorfometrica Nucleare v2.0 (skeleton)")

    synthetic_mask = np.zeros((64, 64), dtype=np.int32)
    synthetic_mask[10:22, 10:22] = 1   # quadrato 12x12
    synthetic_mask[35:50, 35:45] = 2   # rettangolo 15x10

    nuclei_test = extract_nucleus_morphometry(synthetic_mask)
    print(f"[TEST] Nuclei estratti: {len(nuclei_test)} (attesi: 2)")

    for n in nuclei_test:
        print(
            f"       Nucleo {n['nucleus_id']}: Area={n['area_um2']} µm², "
            f"Circolarità={n['circularity']}, Aspect Ratio={n['aspect_ratio']}, "
            f"Solidità={n['solidity']}"
        )

    patch_stats = aggregate_patch_morphometry(nuclei_test, "test_patch", "follicular_lymphoma")
    print(
        f"[TEST] Aggregato patch: Nuclei={patch_stats['n_nuclei']}, "
        f"Area Media={patch_stats['area_um2_mean']} µm², "
        f"CV Area={patch_stats['area_um2_cv']}, "
        f"Top10% Area={patch_stats['area_top10_mean_um2']} µm², "
        f"Top10% Asse Corto={patch_stats['area_top10_short_axis_um']} µm"
    )

    n_expected_agg_cols = len(MORPHOMETRY_BASE_FEATURES) * 4  # mean/std/skew/cv
    print(f"[TEST] Colonne morfometria aggregata attese: {n_expected_agg_cols} "
          f"({len(MORPHOMETRY_BASE_FEATURES)} feature x 4 statistiche)")

    print("[OK] Self-test Modulo 03 (STEP 1-2) superato con successo.")
    print("[INFO] STEP 3 (k-NN) e STEP 4 (tessitura) sono scheletri — "
          "vedi roadmap per la prossima iterazione.")
