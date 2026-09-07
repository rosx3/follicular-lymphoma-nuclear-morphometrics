"""Genera la figura della catena di preparazione per il Capitolo 3 della tesi.

Due righe (un caso FL e uno reattivo, scelti agli estremi della tinta di
partenza) per cinque colonne:

    grezza | Macenko | bilaterale | gaussiana (confronto) | ematossilina+CLAHE

La quarta colonna non fa parte della pipeline: e' il termine di paragone che
mostra perche' il filtro bilaterale sia stato preferito al gaussiano. Usa lo
stesso supporto (kernel 9x9) con ponderazione puramente spaziale.

Uso:  py src/make_figura_preprocessing.py
Esce: img/fase1/figura_preprocessing_tesi.png
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RADICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREZZE = os.path.join(RADICE, "data", "raw")
FASE1 = os.path.join(RADICE, "data", "fase1_preprocessing")
USCITA = os.path.join(RADICE, "img", "fase1", "figura_preprocessing_tesi.png")

# (cartella, nome senza estensione) scelti agli estremi della tinta R-B
CASI = [
    ("follicular_lymphoma", "FL_examples (12)"),
    ("reactive_tissue", "REACTIVE_examples (181)"),
]

# I primi quattro pannelli sono la procedura, nell'ordine in cui viene applicata.
# Il quinto e' fuori procedura e sta in coda, staccato: il filtro gaussiano non
# e' stato impiegato, serve solo da termine di paragone.
INTESTAZIONI = [
    "Immagine grezza",
    "Omogeneizzazione",
    "Filtro bilaterale",
    "Canale ematossilina",
    "Filtro gaussiano\n(confronto)",
]

FUORI_PROCEDURA = 4  # indice della colonna staccata

LATO = 448          # lato del pannello nella figura
BANDA = 96          # altezza della banda delle intestazioni (due righe di testo)
GRONDA = 6          # spazio fra pannelli
STACCO = 58         # spazio aggiuntivo prima della colonna fuori procedura


def carica(percorso, grigio=False):
    modo = "L" if grigio else "RGB"
    return Image.open(percorso).convert(modo).resize((LATO, LATO), Image.LANCZOS)


def pannelli(cartella, nome):
    grezza_p = os.path.join(GREZZE, cartella, nome + ".jpg")
    norm_p = os.path.join(FASE1, cartella, "rgb_normalized", nome + "_norm.png")
    hcan_p = os.path.join(FASE1, cartella, "h_channel", nome + "_hchannel.png")
    for p in (grezza_p, norm_p, hcan_p):
        if not os.path.exists(p):
            raise SystemExit("manca: " + p)

    # bilaterale e gaussiano si calcolano sulla normalizzata, come nella pipeline
    norm = cv2.cvtColor(np.asarray(Image.open(norm_p).convert("RGB")), cv2.COLOR_RGB2BGR)
    bil = cv2.bilateralFilter(norm, d=9, sigmaColor=75, sigmaSpace=75)
    gau = cv2.GaussianBlur(norm, (9, 9), 0)   # stesso supporto, sola componente spaziale
    a_img = lambda m: Image.fromarray(cv2.cvtColor(m, cv2.COLOR_BGR2RGB)).resize(
        (LATO, LATO), Image.LANCZOS)

    return [carica(grezza_p), carica(norm_p), a_img(bil),
            carica(hcan_p, grigio=True).convert("RGB"), a_img(gau)]


def font(dim):
    for f in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(f, dim)
        except OSError:
            continue
    return ImageFont.load_default()


def ascissa(colonna):
    """Ascissa del pannello, con lo stacco davanti alla colonna fuori procedura."""
    x = colonna * (LATO + GRONDA)
    if colonna >= FUORI_PROCEDURA:
        x += STACCO
    return x


def main():
    righe = [pannelli(c, n) for c, n in CASI]
    ncol = len(INTESTAZIONI)
    larghezza = ascissa(ncol - 1) + LATO
    altezza = BANDA + len(righe) * LATO + (len(righe) - 1) * GRONDA
    figura = Image.new("RGB", (larghezza, altezza), "white")
    disegna = ImageDraw.Draw(figura)
    f = font(30)

    for i, testo in enumerate(INTESTAZIONI):
        x = ascissa(i)
        linee = testo.split("\n")
        alt_riga = 36
        y0 = (BANDA - alt_riga * len(linee)) / 2 - 2
        for j, linea in enumerate(linee):
            cassa = disegna.textbbox((0, 0), linea, font=f)
            disegna.text((x + (LATO - (cassa[2] - cassa[0])) / 2, y0 + j * alt_riga),
                         linea, fill="black", font=f)

    for r, riga in enumerate(righe):
        y = BANDA + r * (LATO + GRONDA)
        for c, pan in enumerate(riga):
            figura.paste(pan, (ascissa(c), y))

    # riga verticale che separa la procedura dal termine di paragone
    x_riga = ascissa(FUORI_PROCEDURA) - STACCO // 2
    disegna.line([(x_riga, BANDA - 14), (x_riga, altezza)], fill=(150, 150, 150), width=3)

    os.makedirs(os.path.dirname(USCITA), exist_ok=True)
    figura.save(USCITA, optimize=True)
    print("scritto:", USCITA, figura.size)


if __name__ == "__main__":
    main()
