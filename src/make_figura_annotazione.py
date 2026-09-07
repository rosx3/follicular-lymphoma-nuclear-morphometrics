"""Genera la figura dell'accordo umano per il Capitolo 3 della tesi.

Stessi pannelli di `annotation_agreement.plot_overlay`, ma senza testo cotto
dentro l'immagine: titoli, legenda e numeri stanno nella didascalia LaTeX, dove
si possono correggere e dove seguono il carattere della tesi.

Uso:  py src/make_figura_annotazione.py [nome_immagine]
Esce: img/fase2/annotazione/figura_annotazione_tesi.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from annotation_agreement import (  # noqa: E402
    ANNOTATION_DIR, FASE1_DIR, CATEGORY_DIR,
    _annotation_files, _cellpose_mask, _watershed_mask,
    screen_to_patch, split_certain_and_doubtful,
)

BASE_DIR = Path(__file__).parent.parent
USCITA = BASE_DIR / "img" / "fase2" / "annotazione" / "figura_annotazione_tesi.png"
PREDEFINITA = "FL_examples (15)"


def pannelli(nome: str, destinazione: Path) -> None:
    metadata = json.loads((ANNOTATION_DIR / "immagini_metadata.json").read_text(encoding="utf-8"))
    upscale = metadata["upscale"]

    voce = next((v for v, _, _ in _annotation_files() if v["image_name"] == nome), None)
    if voce is None:
        disponibili = sorted(v["image_name"] for v, _, _ in _annotation_files())
        raise SystemExit(f"{nome!r} non annotata. Disponibili: {disponibili}")

    _, punti_file, dubbi_file = next(
        t for t in _annotation_files() if t[0]["image_name"] == nome
    )
    certi, _ = split_certain_and_doubtful(punti_file, dubbi_file)
    punti = screen_to_patch(certi, upscale)

    maschere = {
        "watershed": _watershed_mask(voce["image_name"], voce["category"]),
        "cellpose": _cellpose_mask(voce["image_name"]),
    }

    rgb_path = FASE1_DIR / CATEGORY_DIR[voce["category"]] / "rgb_normalized" / f"{nome}_norm.png"
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)

    figura, assi = plt.subplots(1, len(maschere), figsize=(7.5 * len(maschere), 7.5))
    assi = np.atleast_1d(assi)

    for asse, (_, maschera) in zip(assi, maschere.items()):
        asse.imshow(rgb)

        colonne = np.clip(np.floor(punti[:, 0]).astype(int), 0, maschera.shape[1] - 1)
        righe = np.clip(np.floor(punti[:, 1]).astype(int), 0, maschera.shape[0] - 1)
        etichette = maschera[righe, colonne]
        presenti, conteggi = np.unique(etichette[etichette > 0], return_counts=True)
        fuse = set(presenti[conteggi > 1].tolist())

        for identificativo in np.unique(maschera):
            if identificativo == 0:
                continue
            contorni, _ = cv2.findContours(
                np.uint8(maschera == identificativo), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            fusa = identificativo in fuse
            for contorno in contorni:
                chiuso = np.vstack([contorno[:, 0, :], contorno[0, 0, :]])
                asse.plot(chiuso[:, 0], chiuso[:, 1],
                          color="#e8590c" if fusa else "#2f9e44",
                          linewidth=1.6 if fusa else 0.7)

        coperti = etichette > 0
        asse.scatter(punti[coperti, 0], punti[coperti, 1],
                     s=14, c="white", edgecolors="black", linewidths=0.4, zorder=3)
        asse.scatter(punti[~coperti, 0], punti[~coperti, 1],
                     s=42, marker="x", c="#e03131", linewidths=1.6, zorder=4)

        asse.set_xticks([])
        asse.set_yticks([])

    figura.tight_layout(pad=0.4)
    destinazione.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destinazione, dpi=200, bbox_inches="tight")
    plt.close(figura)
    print("scritto:", destinazione)


if __name__ == "__main__":
    pannelli(sys.argv[1] if len(sys.argv) > 1 else PREDEFINITA, USCITA)
