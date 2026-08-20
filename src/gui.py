"""
===============================================================================
gui.py — Interfaccia Streamlit
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
===============================================================================
Guscio di widget sopra src/gui_core.py. Qui non c'e' logica scientifica: ogni
numero mostrato arriva dalle stesse funzioni chiamate da run_pipeline.py, cosi'
l'interfaccia non puo' divergere in silenzio dai CSV della tesi.

Tre sezioni:
  1. Esplora dataset   — i cinque stadi della pipeline su una patch gia'
                         elaborata, con i 47 biomarcatori posizionati nella
                         distribuzione della loro classe.
  2. Analizza immagine — un'immagine nuova percorre Fase 1 -> 2 -> 3 dal vivo e
                         viene confrontata con entrambe le classi. Nessuna
                         diagnosi: il classificatore e' la Fase 4, ancora non
                         implementata.
  3. Risultati Fase 3  — figure e test di separabilita' gia' prodotti, il
                         contesto statistico che rende leggibili i percentili.

Avvio:
    streamlit run src/gui.py
===============================================================================
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.color import label2rgb

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from calibration import MICRONS_PER_PIXEL, PATCH_SIZE_PX  # noqa: E402
from gui_core import (  # noqa: E402
    PATCH_FEATURE_COLUMNS,
    available_patches,
    build_normalizer,
    feature_percentile,
    load_patch_images,
    load_reference_image,
    process_image,
)
from naming import CATEGORIES, short_label  # noqa: E402

BASE_DIR = _SRC_DIR.parent
RAW_DIR = BASE_DIR / "data" / "raw"
FASE1_DIR = BASE_DIR / "data" / "fase1_preprocessing"
FASE2_DIR = BASE_DIR / "data" / "fase2_segmentation"
FASE3_DIR = BASE_DIR / "data" / "fase3_features"
IMG_FASE3_DIR = BASE_DIR / "img" / "fase3"

MASTER_CSV = FASE3_DIR / "features_patches_master.csv"
SEPARABILITY_CSV = FASE3_DIR / "separability_tests.csv"

FIGURES = {
    "boxplot_top_features.png": "Biomarcatori piu' discriminanti, FL vs REACTIVE",
    "knn_distribution.png": "Distribuzione delle distanze micro-spaziali k-NN",
    "correlation_heatmap.png": "Correlazione fra i biomarcatori",
    "morphometry_regions_preview.png": "Regioni misurate dalla morfometria nucleare",
}


# ---------------------------------------------------------------------------
# Caricamento dei dati della tesi (in cache: si legge una volta per sessione)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_master_table() -> pd.DataFrame:
    """Matrice 600 x 50 della Fase 3, con i biomarcatori di ogni patch."""
    return pd.read_csv(MASTER_CSV)


@st.cache_data(show_spinner=False)
def load_separability_table() -> pd.DataFrame:
    """Test di separabilita' FL vs REACTIVE con correzione FDR."""
    return pd.read_csv(SEPARABILITY_CSV)


@st.cache_data(show_spinner=False)
def load_catalogue() -> dict[str, list[str]]:
    """Stem delle patch disponibili, per categoria."""
    return available_patches(FASE2_DIR)


@st.cache_resource(show_spinner=False)
def get_normalizer():
    """
    Normalizzatore di Macenko fittato sulla reference image della Fase 1.

    Il fit costa qualche secondo: si esegue una sola volta per processo, ed e'
    la stessa reference usata dalla pipeline, quindi un'immagine caricata dalla
    GUI viene normalizzata esattamente come le 600 del dataset.
    """
    return build_normalizer(load_reference_image(FASE1_DIR, RAW_DIR))


# ---------------------------------------------------------------------------
# Presentazione
# ---------------------------------------------------------------------------
def _mask_preview(mask: np.ndarray) -> np.ndarray:
    """Maschera d'istanza resa visibile: un colore per nucleo, sfondo nero."""
    return label2rgb(mask.astype(np.int32), bg_label=0)


def _show_stages(stages: dict[str, np.ndarray], titles: dict[str, str]) -> None:
    """Mostra affiancati gli stadi della pipeline."""
    for column, (name, caption) in zip(st.columns(len(titles)), titles.items()):
        image = stages[name]
        column.image(
            _mask_preview(image) if name == "mask" else image,
            caption=caption,
            width="stretch",
        )


def _positioned_table(
    values: dict[str, float],
    master: pd.DataFrame,
    references: dict[str, str],
    significant: dict[str, bool],
) -> pd.DataFrame:
    """
    I 47 biomarcatori con il loro valore e la posizione nelle distribuzioni note.

    Args:
        values: biomarcatori della patch o dell'immagine analizzata.
        master: matrice della Fase 3, da cui si ricavano le distribuzioni.
        references: {suffisso di colonna: categoria} da usare come riferimento.
            Una sola voce quando la classe e' nota, entrambe quando non lo e'.
        significant: esito del test di separabilita' per ciascun biomarcatore.
    """
    records = []
    for feature in PATCH_FEATURE_COLUMNS:
        value = float(values[feature])
        record: dict[str, object] = {"biomarcatore": feature, "valore": value}
        for suffix, category in references.items():
            distribution = master.loc[master["category"] == category, feature].to_numpy(float)
            record[f"media_{suffix}"] = float(np.nanmean(distribution))
            record[f"percentile_{suffix}"] = feature_percentile(value, distribution)
        record["significativo"] = bool(significant.get(feature, False))
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sezione 1 — esplorazione di una patch gia' elaborata
# ---------------------------------------------------------------------------
def render_explorer(master: pd.DataFrame, significant: dict[str, bool]) -> None:
    catalogue = load_catalogue()

    st.sidebar.header("Patch da esplorare")
    category = st.sidebar.selectbox(
        "Categoria", CATEGORIES, format_func=short_label, key="categoria"
    )
    stem = st.sidebar.selectbox("Patch", catalogue[category], key="patch")

    st.subheader(f"{stem} — {short_label(category)}")
    st.caption(
        "Gli stadi della pipeline, dall'immagine grezza alla segmentazione dei nuclei."
    )

    images = load_patch_images(stem, category, RAW_DIR, FASE1_DIR, FASE2_DIR)
    _show_stages(
        images,
        {
            "raw": "1. Grezza (H&E)",
            "normalized": "2. Macenko + bilaterale",
            "h_channel": "3. Canale H + CLAHE",
            "overlay": "4. Contorni dei nuclei",
            "mask": "5. Maschera d'istanza",
        },
    )

    row = master.loc[master["image_name"] == stem]
    if row.empty:
        st.error(f"'{stem}' non compare in {MASTER_CSV.name}: rieseguire la Fase 3.")
        return

    values = row.iloc[0]
    st.metric("Nuclei segmentati", int(values["n_nuclei"]))

    st.subheader("Biomarcatori")
    st.caption(
        "`percentile_classe` dice quanto la patch e' tipica rispetto alla propria "
        "classe: 50 significa perfettamente mediana, valori agli estremi indicano "
        "una patch anomala. `significativo` riporta l'esito del test FL vs "
        "REACTIVE con correzione FDR."
    )
    st.dataframe(
        _positioned_table(values, master, {"classe": category}, significant),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Sezione 2 — analisi di un'immagine nuova
# ---------------------------------------------------------------------------
def render_analyzer(master: pd.DataFrame, significant: dict[str, bool]) -> None:
    st.warning(
        "Questa sezione **non fornisce una diagnosi**: misura biomarcatori e li "
        "confronta con le distribuzioni delle due classi. Il classificatore e' la "
        "Fase 4 del progetto, non ancora implementata.",
        icon="⚠️",
    )
    st.caption(
        "L'immagine caricata percorre le stesse Fasi 1 -> 2 -> 3 delle 600 patch "
        "del dataset, chiamando le identiche funzioni della pipeline."
    )
    st.info(
        "**Scarto residuo.** L'analisi dal vivo ricalcola la Fase 1 in memoria, "
        "mentre il dataset e' stato costruito rileggendo i PNG normalizzati salvati "
        "su disco: il doppio passaggio di quantizzazione rende i due percorsi non "
        "identici. Rielaborando patch gia' nel dataset lo scarto sul numero di "
        "nuclei e' risultato di mediana 0.0% ed estremi -1.9% / +1.8% (16 patch). "
        "Va tenuto presente leggendo i percentili, che sono calcolati su "
        "distribuzioni prodotte dall'altro percorso.",
        icon="ℹ️",
    )

    uploaded = st.file_uploader(
        "Immagine istologica H&E", type=["png", "jpg", "jpeg", "tif", "tiff"], key="upload"
    )
    if uploaded is None:
        return

    decoded = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        st.error(f"'{uploaded.name}' non e' un'immagine leggibile.")
        return
    image_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    # La calibrazione spaziale vale per patch della dimensione attesa: su
    # immagini diverse le misure in um non sono confrontabili con il dataset.
    if image_rgb.shape[:2] != (PATCH_SIZE_PX, PATCH_SIZE_PX):
        st.warning(
            f"L'immagine e' {image_rgb.shape[1]}x{image_rgb.shape[0]} px, mentre la "
            f"pipeline e' calibrata su patch {PATCH_SIZE_PX}x{PATCH_SIZE_PX} a "
            f"{MICRONS_PER_PIXEL} um/px. Le misure in um restano calcolabili, ma il "
            "confronto con le distribuzioni del dataset non e' piu' garantito.",
            icon="📏",
        )

    with st.spinner("Normalizzazione, segmentazione ed estrazione dei biomarcatori..."):
        result = process_image(image_rgb, get_normalizer())

    st.subheader(uploaded.name)
    _show_stages(
        {"raw": image_rgb, **result},
        {
            "raw": "1. Caricata",
            "normalized": "2. Macenko",
            "h_channel": "3. Canale H + CLAHE",
            "overlay": "4. Contorni dei nuclei",
            "mask": "5. Maschera d'istanza",
        },
    )

    st.metric("Nuclei segmentati", len(result["nuclei"]))
    if not result["nuclei"]:
        st.error(
            "Nessun nucleo segmentato: l'immagine non sembra una patch istologica "
            "H&E alla scala attesa."
        )
        return

    st.subheader("Biomarcatori a confronto con le due classi")
    st.caption(
        "Per ogni biomarcatore, il percentile occupato nella distribuzione di "
        "ciascuna classe. Un valore molto lontano da 50 in una sola delle due "
        "colonne indica affinita' con l'altra."
    )
    st.dataframe(
        _positioned_table(
            result["features"],
            master,
            {short_label(category): category for category in CATEGORIES},
            significant,
        ),
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Sezione 3 — risultati statistici della Fase 3
# ---------------------------------------------------------------------------
def render_results(separability: pd.DataFrame) -> None:
    n_significant = int(separability["significant"].sum())
    best = separability.sort_values("p_fdr").iloc[0]

    left, right = st.columns(2)
    left.metric("Biomarcatori significativi", f"{n_significant} / {len(separability)}")
    right.metric("Piu' discriminante", best["feature"], f"p (FDR) = {best['p_fdr']:.1e}")

    st.subheader("Test di separabilita' FL vs REACTIVE")
    st.caption(
        "Mann-Whitney U o t-test di Welch a seconda della normalita' osservata, "
        "con correzione FDR di Benjamini-Hochberg."
    )
    st.dataframe(
        separability.sort_values("p_fdr"),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Figure")
    for column, (filename, caption) in zip(st.columns(2), list(FIGURES.items())[:2]):
        column.image(str(IMG_FASE3_DIR / filename), caption=caption)
    for column, (filename, caption) in zip(st.columns(2), list(FIGURES.items())[2:]):
        column.image(str(IMG_FASE3_DIR / filename), caption=caption)


# ---------------------------------------------------------------------------
# Applicazione
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Biomarcatori nucleari — FL vs Tessuto Reattivo",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Quantificazione citomorfometrica e spaziale")
    st.caption(
        "Linfoma follicolare vs tessuto linfoide reattivo — approccio white-box "
        "su 600 patch H&E e 94.042 nuclei."
    )

    missing = [path.name for path in (MASTER_CSV, SEPARABILITY_CSV) if not path.exists()]
    if missing:
        st.error(
            f"Dati della Fase 3 mancanti ({', '.join(missing)}). Eseguire prima "
            "`python src/run_pipeline.py` e `python src/feature_analysis.py`."
        )
        return

    master = load_master_table()
    separability = load_separability_table()
    significant = separability.set_index("feature")["significant"].to_dict()

    explorer, analyzer, results = st.tabs(
        ["Esplora dataset", "Analizza immagine", "Risultati Fase 3"]
    )
    with explorer:
        render_explorer(master, significant)
    with analyzer:
        render_analyzer(master, significant)
    with results:
        render_results(separability)


# Streamlit esegue lo script come "__main__": la guardia lascia il modulo
# importabile dai test senza far partire l'intera interfaccia.
if __name__ == "__main__":
    main()
