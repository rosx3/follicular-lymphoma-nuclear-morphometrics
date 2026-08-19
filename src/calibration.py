"""
===============================================================================
calibration.py — Calibrazione Spaziale del Dataset
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
===============================================================================
Unica fonte di verita' per la conversione da pixel a unita' fisiche. Ogni
grandezza in micron o micron quadri prodotta dalla pipeline deriva da qui.

PROVENIENZA DEL VALORE
----------------------
Lo studio sorgente (Carreras J. et al., 2025, Cancers 17(15):2428) dichiara:

  - Scanner: Hamamatsu NanoZoomer S360 C13220-01
  - Esportazione delle patch via NDP.view2 "at 200x magnification and 150 dpi"
  - Patch di 224 x 224 x 3 px

L'articolo NON pubblica un valore di micron per pixel, NON indica la dimensione
fisica del campo visivo e NON menziona mai un obiettivo 40x.

Nella notazione istopatologica convenzionale l'ingrandimento e' riportato come
prodotto obiettivo x oculare (10x): i 200x dichiarati corrispondono quindi a un
obiettivo 20x, cioe' meta' della risoluzione nativa del NanoZoomer S360
(0.23 um/px a obiettivo 40x, pari a 400x totali). Da cui il valore adottato.
Coerentemente, le figure dell'articolo esportate a 400x corrispondono
all'obiettivo 40x nativo.

VERIFICHE INDIPENDENTI (vedi reports/fase3_report.md 3.4)
---------------------------------------------------------
Due controlli che non usano la calibrazione per essere calcolati, e che quindi
la possono validare:

  - Densita' nucleare: risulta ~14.500 nuclei/mm2, dentro l'intervallo di
    letteratura per il tessuto linfoide (10.000-20.000/mm2). Con la
    calibrazione precedente risultava ~58.000/mm2, fisicamente impossibile.
  - Diametro nucleare medio: risulta ~4.96 um misurato, che corretto per la
    sotto-copertura del Watershed (Dice 0.637 rispetto alla Ground Truth)
    porta a ~6.2 um, valore atteso per un nucleo linfocitario.

AVVERTENZA
----------
Il valore resta una DEDUZIONE dalle condizioni di esportazione dichiarate, non
un dato pubblicato dagli autori. Una conferma definitiva richiederebbe di
contattare gli autori dello studio sorgente o di misurare una struttura di
dimensione nota. Le grandezze adimensionali della pipeline (conteggi,
frazioni di area, circolarita', eccentricita', solidita', aspect ratio,
coefficienti di variazione, tessitura) non dipendono da questo valore.
===============================================================================
"""

# Dimensione del lato di un pixel sul preparato, in micron.
MICRONS_PER_PIXEL: float = 0.46

# Area di un pixel, in micron quadri. Una revisione della scala agisce
# linearmente sulle lunghezze e quadraticamente sulle aree.
PIXEL_AREA_UM2: float = MICRONS_PER_PIXEL**2

# Geometria della patch (fissata dall'input layer della rete nello studio sorgente).
PATCH_SIZE_PX: int = 224
PATCH_SIDE_UM: float = PATCH_SIZE_PX * MICRONS_PER_PIXEL
PATCH_AREA_UM2: float = PATCH_SIDE_UM**2


def px_to_um(length_px: float) -> float:
    """Converte una lunghezza da pixel a micron."""
    return length_px * MICRONS_PER_PIXEL


def px2_to_um2(area_px: float) -> float:
    """Converte un'area da pixel quadri a micron quadri."""
    return area_px * PIXEL_AREA_UM2
