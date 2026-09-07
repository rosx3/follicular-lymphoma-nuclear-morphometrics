"""Genera la figura dei contributi SHAP per il Capitolo 4.

Sciame di punti: una riga per biomarcatore, un punto per patch. La posizione
orizzontale e' il contributo alla decisione, il colore il valore assunto dal
biomarcatore su quella patch. E' l'unica figura della Fase 4 che una tabella non
sostituisce, perche' mostra insieme il verso del contributo e la sua dispersione.

Rispetto alla figura di lavoro `img/fase4/shap_summary.png` cambia tre cose: i
nomi sono quelli usati nella tesi e non gli identificatori di colonna, la scala
di colore e' monocroma per non confondersi con i due colori delle classi, e i
biomarcatori mostrati sono dieci invece di venti.

Uso:  py src/make_figura_shap.py
Esce: img/fase4/figura_shap_tesi.png
"""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATI = os.path.join(RADICE, "data", "fase3_features", "features_patches_master.csv")
MODELLO = os.path.join(RADICE, "data", "fase4_classification", "best_model.joblib")
USCITA = os.path.join(RADICE, "img", "fase4", "figura_shap_tesi.png")

QUANTI = 10

NOMI = {
    "lbp_entropy": "Entropia dei modelli binari locali",
    "hchannel_mean": "Intensità media di ematossilina",
    "solidity_mean": "Solidità media",
    "glcm_contrast": "Contrasto della cromatina",
    "knn1_dist_mean_um": "Distanza media al primo vicino",
    "n_nuclei": "Numero di nuclei",
    "hchannel_std": "Dispersione dell'intensità di ematossilina",
    "glcm_homogeneity": "Omogeneità della cromatina",
    "minor_axis_um_skew": "Asimmetria dell'asse minore",
    "major_axis_um_cv": "Variabilità dell'asse maggiore",
    "aspect_ratio_skew": "Asimmetria del rapporto d'aspetto",
    "area_um2_mean": "Area nucleare media",
    "area_top10_mean_um2": "Area media del decimo più grande",
    "nuclear_area_fraction": "Frazione di area occupata dai nuclei",
    "solidity_std": "Dispersione della solidità",
}

SCALA = LinearSegmentedColormap.from_list(
    "biomarcatore", ["#dbe3ea", "#7d94a8", "#2f4858", "#1b2b36"]
)


def _valori_shap(X: np.ndarray, modello) -> np.ndarray:
    """Valori di Shapley per la classe linfoma follicolare (target = 1)."""
    import shap

    stimatore = modello.steps[-1][1] if hasattr(modello, "steps") else modello
    valori = np.asarray(shap.TreeExplainer(stimatore).shap_values(X))
    if valori.ndim == 3:
        valori = valori[:, :, 1]
    return valori


def _scarti_verticali(x: np.ndarray, semiampiezza: float = 0.36) -> np.ndarray:
    """Distribuisce i punti in verticale dove si accalcano, come in uno sciame."""
    bordi = np.linspace(x.min(), x.max(), 60) if x.max() > x.min() else np.array([x.min(), x.max() + 1])
    cestello = np.clip(np.digitize(x, bordi) - 1, 0, len(bordi) - 2)
    conteggi = np.bincount(cestello, minlength=len(bordi) - 1)
    massimo = max(int(conteggi.max()), 1)

    scarti = np.zeros_like(x, dtype=float)
    for indice in np.unique(cestello):
        posizioni = np.where(cestello == indice)[0]
        quanti = len(posizioni)
        larghezza = semiampiezza * np.sqrt(quanti / massimo)
        if quanti == 1:
            scarti[posizioni] = 0.0
        else:
            scarti[posizioni] = np.linspace(-larghezza, larghezza, quanti)
    return scarti


def main():
    tabella = pd.read_csv(DATI)
    fascio = joblib.load(MODELLO)
    modello, biomarcatori = fascio["model"], fascio["features"]

    X = tabella[biomarcatori].to_numpy(dtype=float)
    valori = _valori_shap(X, modello)

    importanza = np.abs(valori).mean(axis=0)
    scelti = np.argsort(importanza)[::-1][:QUANTI]

    figura, asse = plt.subplots(figsize=(11, 6.2))
    asse.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)

    for riga, colonna in enumerate(scelti):
        y = QUANTI - 1 - riga
        contributi = valori[:, colonna]
        grandezza = X[:, colonna]
        # Il colore segue il rango, non il valore: un solo nucleo anomalo non
        # schiaccia l'intera scala.
        ranghi = pd.Series(grandezza).rank(pct=True).to_numpy()
        asse.scatter(contributi, y + _scarti_verticali(contributi),
                     c=ranghi, cmap=SCALA, norm=Normalize(0, 1),
                     s=9, linewidths=0, alpha=0.85)

    asse.set_yticks(range(QUANTI))
    asse.set_yticklabels([NOMI.get(biomarcatori[c], biomarcatori[c])
                          for c in scelti][::-1], fontsize=10)
    asse.set_ylim(-0.7, QUANTI - 0.3)
    asse.set_xlabel("Contributo alla decisione (valore di Shapley)", fontsize=10)
    asse.tick_params(axis="x", labelsize=9)
    asse.grid(axis="x", alpha=0.2, linewidth=0.6)
    asse.set_axisbelow(True)
    for lato in ("top", "right", "left"):
        asse.spines[lato].set_visible(False)

    limite = asse.get_xlim()
    asse.text(limite[0], -1.35, "verso tessuto reattivo", fontsize=9,
              ha="left", va="center", style="italic")
    asse.text(limite[1], -1.35, "verso linfoma follicolare", fontsize=9,
              ha="right", va="center", style="italic")

    barra = figura.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=SCALA),
                            ax=asse, pad=0.02, fraction=0.03, ticks=[0.02, 0.98])
    barra.ax.set_yticklabels(["basso", "alto"], fontsize=9)
    barra.set_label("valore del biomarcatore", fontsize=9)
    barra.outline.set_visible(False)

    figura.tight_layout(pad=1.2)
    os.makedirs(os.path.dirname(USCITA), exist_ok=True)
    figura.savefig(USCITA, dpi=220, bbox_inches="tight")
    plt.close(figura)
    print("scritto:", USCITA)


if __name__ == "__main__":
    main()
