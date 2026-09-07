"""Genera la figura dei biomarcatori piu' discriminanti per il Capitolo 4.

Sei riquadri, uno per biomarcatore, con le due classi affiancate. Nessun testo
cotto dentro l'immagine oltre ai nomi degli assi: titolo, valori di probabilita'
e lettura stanno nella didascalia LaTeX.

Rispetto alla figura di Fase 3 cambia tre cose: i nomi sono quelli usati nella
tesi e non gli identificatori di colonna, spariscono i valori di probabilita'
(la tabella della 4.2 li riporta gia'), e sparisce il pannello della densita'
nucleare, collineare con il numero di nuclei.

Uso:  py src/make_figura_separabilita.py
Esce: img/fase3/figura_separabilita_tesi.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATI = os.path.join(RADICE, "data", "fase3_features", "features_patches_master.csv")
USCITA = os.path.join(RADICE, "img", "fase3", "figura_separabilita_tesi.png")

# (colonna, nome nella tesi, unita' di misura)
PANNELLI = [
    ("lbp_entropy", "Entropia dei modelli\nbinari locali", "bit"),
    ("hchannel_mean", "Intensità media\ndi ematossilina", "livelli [0-255]"),
    ("nuclear_area_fraction", "Frazione di area\noccupata dai nuclei", "adimensionale"),
    ("knn3_dist_mean_um", "Distanza media\nai tre vicini", "µm"),
    ("n_nuclei", "Numero di nuclei", "conteggio"),
    ("glcm_homogeneity", "Omogeneità\ndella cromatina", "adimensionale"),
]

CLASSI = [("follicular_lymphoma", "Linfoma\nfollicolare", "#c9736b"),
          ("reactive_tissue", "Tessuto\nreattivo", "#5b84a8")]


def main():
    dati = pd.read_csv(DATI)
    figura, assi = plt.subplots(2, 3, figsize=(11, 6.4))

    for asse, (colonna, nome, unita) in zip(assi.ravel(), PANNELLI):
        gruppi = [dati.loc[dati["category"] == c, colonna].dropna() for c, _, _ in CLASSI]
        riquadri = asse.boxplot(gruppi, patch_artist=True, widths=0.55,
                                medianprops=dict(color="black", linewidth=1.4),
                                flierprops=dict(marker=".", markersize=3,
                                                markerfacecolor="grey",
                                                markeredgecolor="none", alpha=0.5))
        for corpo, (_, _, colore) in zip(riquadri["boxes"], CLASSI):
            corpo.set_facecolor(colore)
            corpo.set_alpha(0.85)
            corpo.set_edgecolor("black")
            corpo.set_linewidth(0.8)

        asse.set_title(nome, fontsize=10.5)
        asse.set_ylabel(unita, fontsize=9)
        asse.set_xticks([1, 2])
        asse.set_xticklabels([e for _, e, _ in CLASSI], fontsize=9)
        asse.tick_params(axis="y", labelsize=8)
        asse.grid(axis="y", alpha=0.25, linewidth=0.6)
        asse.set_axisbelow(True)
        for lato in ("top", "right"):
            asse.spines[lato].set_visible(False)

    figura.tight_layout(pad=1.2, h_pad=2.0)
    os.makedirs(os.path.dirname(USCITA), exist_ok=True)
    figura.savefig(USCITA, dpi=220, bbox_inches="tight")
    plt.close(figura)
    print("scritto:", USCITA)


if __name__ == "__main__":
    main()
