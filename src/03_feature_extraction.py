"""
===============================================================================
Modulo 03: Estrazione Biomarcatori Citomorfometrici e Spaziali
Tesi: Classificazione Linfoma Follicolare vs Tessuto Reattivo
===============================================================================
Questo modulo gestisce l'estrazione di:
 1. Morfologia nucleare (Area in µm², Perimetro in µm, Circolarità, Eccentricità, Solidità)
 2. Pleomorfismo (Deviazione Standard e Skewness intra-patch)
 3. Densità cellulare (% occupazione nucleare su 51.5 x 51.5 µm²)
 4. Distanze inter-nucleari k-NN (k=1, 3, 5 in µm)
 5. Grafi di prossimità spaziale (Delaunay, Minimum Spanning Tree - MST)
 6. Tessitura e Colore (GLCM Haralick, LBP, CIE-LAB)
===============================================================================
"""

import numpy as np

# Calibrazione spaziale (Hamamatsu NanoZoomer S360 - 40x)
MICRONS_PER_PIXEL = 0.23  # µm / px
PIXEL_AREA_MICRONS2 = MICRONS_PER_PIXEL ** 2  # 0.0529 µm² / px²

def extract_cytomorphometric_features(labeled_mask):
    """
    Estrae metriche di dimensione, forma e pleomorfismo nucleare.
    Tutte le distanze sono convertite in µm ed aree in µm².
    """
    pass

def extract_spatial_graph_features(centroids_px):
    """
    Calcola k-NN, Tassellatura di Delaunay e Minimum Spanning Tree (MST)
    sui centroidi dei nuclei.
    """
    pass

if __name__ == '__main__':
    print(f"[INFO] Modulo Feature Extraction inizializzato. Scala: {MICRONS_PER_PIXEL} µm/px")
