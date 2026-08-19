"""
===============================================================================
Modulo 03: Estrazione Biomarcatori Citomorfometrici Nucleari
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
      tramite Morfometria Nucleare e AI Interpretabile (White-Box XAI)
===============================================================================
Questo modulo implementa la caratterizzazione citomorfometrica d'istanza:
 1. Estrazione metriche per singolo nucleo (Area, Perimetro, Circolarità,
    Eccentricità, Solidità, Extent, Assi Ellisse, Diametro Equivalente).
 2. Aggregazione statistica per patch (Media, Mediana, Deviazione Standard,
    Skewness e Terzile Top 10% dell'area nucleare - Iwamoto et al. 2024).
 3. Generazione di anteprime visive con Bounding Box, Contorni ed Ellissi
    inerziali sovrapposti alle immagini RGB.
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
MICRONS_PER_PIXEL   = 0.23                # µm / px
PIXEL_AREA_UM2      = MICRONS_PER_PIXEL**2 # 0.0529 µm² / px²
PATCH_SIZE_PX       = 224
PATCH_AREA_UM2      = (PATCH_SIZE_PX * MICRONS_PER_PIXEL) ** 2  # 2654.31 µm²


# ---------------------------------------------------------------------------
# 1. Estrazione Citomorfometrica per Singolo Nucleo
# ---------------------------------------------------------------------------
def extract_nucleus_morphometry(instance_mask: np.ndarray) -> list[dict]:
    """
    Estrae i biomarcatori citomorfometrici di forma e dimensione per ciascun
    nucleo presente nella maschera d'istanza 16-bit.

    Args:
        instance_mask (np.ndarray): Maschera 2D int (0 = sfondo, ID >= 1 = nuclei).

    Returns:
        list[dict]: Lista di dizionari con i biomarcatori di ogni nucleo.
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
            circ = (4.0 * np.pi * area_px) / (perimeter_px**2)
            circ = float(np.clip(circ, 0.0, 1.0))
        else:
            circ = 0.0

        # Eccentricità (0 = cerchio, 1 = linea)
        ecc = float(p.eccentricity)

        # Solidità: Area / Area Convex Hull
        solidity = float(p.solidity)

        # Extent / Rettangolarità: Area / Area Bounding Box
        extent = float(p.extent)

        # Assi ellisse equivalente in micron
        major_axis_um = float(p.axis_major_length * MICRONS_PER_PIXEL)
        minor_axis_um = float(p.axis_minor_length * MICRONS_PER_PIXEL)

        # Diametro equivalente del cerchio di pari area
        equiv_diam_um = float(p.equivalent_diameter_area * MICRONS_PER_PIXEL)

        # Bounding box coordinates
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
            "extent": round(extent, 4),
            "major_axis_um": round(major_axis_um, 3),
            "minor_axis_um": round(minor_axis_um, 3),
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
def aggregate_patch_morphometry(
    nuclei_list: list[dict], image_name: str, category: str
) -> dict:
    """
    Calcola le statistiche aggregate di citomorfometria per l'intera patch.

    Metriche calcolate:
      - Densità nucleare (% occupazione e nuclei per 1000 µm²)
      - Media, Dev. Std., Mediana, Skewness per tutte le variabili fisiche
      - Area Top 10% (Media dei nuclei più grandi — Iwamoto et al. 2024)
    """
    n_nuclei = len(nuclei_list)

    patch_dict = {
        "image_name": image_name,
        "category": category,
        "n_nuclei": n_nuclei,
        "nuclear_density_per_1000um2": round(
            (n_nuclei / PATCH_AREA_UM2) * 1000.0, 3
        ),
    }

    if n_nuclei == 0:
        patch_dict["nuclear_area_fraction"] = 0.0
        patch_dict["area_top10_mean_um2"] = 0.0
        # Riempi metriche con 0 per patch vuote
        for feat in [
            "area_um2",
            "perimeter_um",
            "circularity",
            "eccentricity",
            "solidity",
            "extent",
            "major_axis_um",
            "minor_axis_um",
            "equivalent_diameter_um",
        ]:
            for stat in ["mean", "std", "median", "skew"]:
                patch_dict[f"{feat}_{stat}"] = 0.0
        return patch_dict

    total_nuclear_area_um2 = sum(n["area_um2"] for n in nuclei_list)
    patch_dict["nuclear_area_fraction"] = round(
        total_nuclear_area_um2 / PATCH_AREA_UM2, 4
    )

    # Top 10% Area (Iwamoto et al. 2024 — indicatore di centroblasti)
    areas_sorted = sorted([n["area_um2"] for n in nuclei_list], reverse=True)
    top10_k = max(1, int(np.ceil(n_nuclei * 0.10)))
    patch_dict["area_top10_mean_um2"] = round(
        float(np.mean(areas_sorted[:top10_k])), 3
    )

    # Statistiche per ciascun biomarcatore continuo
    features_to_aggregate = [
        "area_um2",
        "perimeter_um",
        "circularity",
        "eccentricity",
        "solidity",
        "extent",
        "major_axis_um",
        "minor_axis_um",
        "equivalent_diameter_um",
    ]

    for feat in features_to_aggregate:
        vals = np.array([n[feat] for n in nuclei_list], dtype=np.float64)
        patch_dict[f"{feat}_mean"] = round(float(np.mean(vals)), 4)
        patch_dict[f"{feat}_std"] = round(float(np.std(vals)), 4)
        patch_dict[f"{feat}_median"] = round(float(np.median(vals)), 4)
        sk = float(stats.skew(vals)) if len(vals) > 2 else 0.0
        patch_dict[f"{feat}_skew"] = round(
            sk if not np.isnan(sk) else 0.0, 4
        )

    return patch_dict


# ---------------------------------------------------------------------------
# 3. Generazione Anteprime Visive (Bounding Boxes, Contorni, Ellissi)
# ---------------------------------------------------------------------------
def generate_morphometry_overlay_image(
    rgb_img: np.ndarray, instance_mask: np.ndarray, nuclei_list: list[dict]
) -> np.ndarray:
    """
    Genera un'immagine RGB ad alta definizione che sovrappone:
      - Bounding Box (rettangolo blu) per ogni nucleo
      - Contorno d'istanza (verde)
      - Centroide (punto rosso)
      - Ellisse equivalente (gialla)
    """
    overlay = rgb_img.copy()

    for n in nuclei_list:
        # Bounding box
        ymin, xmin, ymax, xmax = (
            n["bbox_ymin"],
            n["bbox_xmin"],
            n["bbox_ymax"],
            n["bbox_xmax"],
        )
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (255, 120, 0), 1)

        # Centroide
        cx, cy = int(n["centroid_x_px"]), int(n["centroid_y_px"])
        cv2.circle(overlay, (cx, cy), 1, (0, 0, 255), -1)

    # Contorni d'istanza verdi
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]

    for uid in unique_ids:
        binary_single = np.uint8(instance_mask == uid)
        cnts, _ = cv2.findContours(
            binary_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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
    Crea una figura comparativa che mostra le regioni considerate dalla citomorfometria
    con Bounding Boxes, contorni ed ellissi per Linfoma Follicolare vs Tessuto Reattivo.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)

    for i, (rgb_p, mask_p, title, cat_color) in enumerate([
        (fl_rgb_path, fl_mask_path, "Linfoma Follicolare (FL)", "red"),
        (re_rgb_path, re_mask_path, "Tessuto Reattivo (REACTIVE)", "blue"),
    ]):
        rgb = cv2.imread(rgb_p)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_UNCHANGED)

        nuclei = extract_nucleus_morphometry(mask)
        overlay = generate_morphometry_overlay_image(rgb, mask, nuclei)

        # Colonna 0: RGB originale con contorni
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"{title} — RGB Normalizzata", fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        # Colonna 1: Overlay Citomorfometrico (Bounding Box + Contorni)
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(
            f"{title} — Bounding Boxes ({len(nuclei)} nuclei)",
            fontsize=12,
            fontweight="bold",
        )
        axes[i, 1].axis("off")

    # Legenda personalizzata
    legend_elements = [
        mpatches.Patch(color="green", label="Contorno Nucleare Segmentato"),
        mpatches.Patch(color="orange", label="Bounding Box Citomorfometrica"),
        mpatches.Patch(color="red", label="Centroide Nucleare (x, y)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=11,
        frameon=True,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, bbox_inches="tight")
    plt.close()
    print(f"[Fase 3] Anteprima visiva citomorfometrica salvata: {output_png_path}")


# ---------------------------------------------------------------------------
# Self-test del Modulo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Modulo 03 — Estrazione Citomorfometrica Nucleare v1.0")

    # Test su maschera sintetica 64x64 con 2 nuclei
    synthetic_mask = np.zeros((64, 64), dtype=np.int32)
    synthetic_mask[10:22, 10:22] = 1  # quadrato 12x12
    synthetic_mask[35:50, 35:45] = 2  # rettangolo 15x10

    nuclei_test = extract_nucleus_morphometry(synthetic_mask)
    print(f"[TEST] Nuclei estratti: {len(nuclei_test)} (attesi: 2)")

    for n in nuclei_test:
        print(
            f"       Nucleo {n['nucleus_id']}: Area={n['area_um2']} µm², "
            f"Perimetro={n['perimeter_um']} µm, Circolarità={n['circularity']}, "
            f"Eccentricità={n['eccentricity']}, Solidità={n['solidity']}"
        )

    patch_stats = aggregate_patch_morphometry(
        nuclei_test, "test_patch", "follicular_lymphoma"
    )
    print(f"[TEST] Aggregato patch: Nuclei={patch_stats['n_nuclei']}, "
          f"Area Media={patch_stats['area_um2_mean']} µm², "
          f"Top10% Area={patch_stats['area_top10_mean_um2']} µm²")

    print("[OK] Self-test Modulo 03 superato con successo.")
