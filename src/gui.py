"""
===============================================================================
gui.py — Interfaccia Streamlit
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
===============================================================================
Guscio di widget sopra src/gui_core.py. Qui non c'e' logica scientifica: ogni
numero mostrato arriva dalle stesse funzioni chiamate da run_pipeline.py, cosi'
l'interfaccia non puo' divergere in silenzio dai CSV della tesi.

Quattro sezioni:
  1. Esplora dataset   — i cinque stadi della pipeline su una patch gia'
                         elaborata, i 47 biomarcatori posizionati nella
                         distribuzione della loro classe e la predizione
                         fuori-piega del modello.
  2. Analizza immagine — un'immagine nuova percorre Fase 1 -> 2 -> 3 dal vivo,
                         viene classificata e la decisione viene spiegata.
  3. Spiegabilita'     — l'analisi dei risultati della Fase 4: la forbice fra
                         validazione ottimistica e conservativa, quali
                         biomarcatori decidono e in che direzione (dichiarata
                         solo dove l'effetto e' monotono), e la spiegazione
                         locale di un caso a scelta, calcolata dal vivo.
  4. Risultati Fase 3  — figure e test di separabilita' gia' prodotti, il
                         contesto statistico che rende leggibili i percentili.

Avvio:
    streamlit run src/gui.py
===============================================================================
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from skimage.color import label2rgb

# Un'app Streamlit non ha un display, e soprattutto esegue lo script in un thread
# separato. Col backend predefinito di questa macchina (Tk) la figura del
# waterfall nasce in quel thread e viene distrutta dal thread principale alla
# chiusura: Tk aborta il processo con "Tcl_AsyncDelete: async handler deleted by
# the wrong thread", uccidendo la suite di test invece di fallire. Agg non ha
# alcuna GUI e non ha il problema. Va impostato PRIMA del primo import di pyplot.
matplotlib.use("Agg")

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from calibration import MICRONS_PER_PIXEL, PATCH_SIZE_PX  # noqa: E402
from gui_core import (  # noqa: E402
    PATCH_FEATURE_COLUMNS,
    available_patches,
    build_normalizer,
    explain_patch,
    feature_percentile,
    load_classifier,
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

FASE4_DIR = BASE_DIR / "data" / "fase4_classification"
IMG_FASE4_DIR = BASE_DIR / "img" / "fase4"
METRICS_CSV = FASE4_DIR / "metrics_by_model.csv"
SENSITIVITY_CSV = FASE4_DIR / "block_size_sensitivity.csv"
REDUCTION_CSV = FASE4_DIR / "feature_reduction.csv"

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


@st.cache_data(show_spinner=False)
def load_fase4_table(file_name: str) -> pd.DataFrame:
    """Una delle tabelle prodotte dalla Fase 4, letta una volta per sessione."""
    return pd.read_csv(FASE4_DIR / file_name)


@st.cache_resource(show_spinner=False)
def get_classifier():
    """
    Modello della Fase 4, caricato una volta per processo.

    E' addestrato sui 33 biomarcatori sopravvissuti alla riduzione delle
    ridondanze: l'artefatto porta con se' l'elenco, e la selezione avviene per
    nome, mai per posizione.
    """
    return load_classifier(FASE4_DIR)


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


def _render_probability(probability: float, truth: int | None = None) -> None:
    """Probabilita' di linfoma follicolare, con l'esito quando la verita' e' nota."""
    predicted = "FL" if probability >= 0.5 else "REACTIVE"
    confidence = probability if probability >= 0.5 else 1 - probability

    left, middle, right = st.columns(3)
    left.metric("Probabilita' di linfoma follicolare", f"{probability:.1%}")
    middle.metric("Classe predetta", predicted, f"confidenza {confidence:.0%}")
    if truth is not None:
        actual = "FL" if truth == 1 else "REACTIVE"
        correct = (probability >= 0.5) == (truth == 1)
        right.metric("Classe reale", actual, "corretta" if correct else "SBAGLIATA",
                     delta_color="normal" if correct else "inverse")
    st.progress(float(probability))


def _render_waterfall(explanation, top_n: int = 12) -> None:
    """
    Contributi che hanno spostato la decisione su questa patch.

    E' la spiegazione locale: non "quali biomarcatori contano in generale", ma
    "quali hanno deciso QUI, e di quanto". La somma dei contributi piu' il valore
    atteso ricostruisce esattamente l'uscita del modello.
    """
    import matplotlib.pyplot as plt

    top = explanation.contributions.head(top_n).iloc[::-1]
    colors = ["#c0392b" if c > 0 else "#2471a3" for c in top["contribution"]]
    labels = [f"{row.feature} = {row.value:.3g}" for row in top.itertuples()]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(top) + 1.2))
    ax.barh(range(len(top)), top["contribution"], color=colors)
    ax.set_yticks(range(len(top)), labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("contributo alla decisione  (rosso -> FL, blu -> REACTIVE)")
    ax.set_title("Perche' il modello ha deciso cosi'")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"Valore atteso {explanation.expected_value:+.3f} + somma dei contributi "
        f"= {explanation.raw_output:+.3f} (log-odds). E' la proprieta' che rende "
        "questo grafico una spiegazione e non un'illustrazione: e' verificata da un test."
    )


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

    _render_out_of_fold_prediction(stem, int(values["target"]))

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


def _render_out_of_fold_prediction(stem: str, truth: int) -> None:
    """
    Predizione del modello per una patch del dataset.

    Si mostra quella FUORI-PIEGA registrata dalla Fase 4, non una predizione
    calcolata al momento: il modello finale e' stato addestrato su tutte le 600
    patch, quindi rieseguirlo su una di esse restituirebbe un numero ottimistico
    e privo di significato. La predizione fuori-piega viene invece da un modello
    che quella patch non l'aveva mai vista.
    """
    if not (FASE4_DIR / "out_of_fold_predictions.csv").exists():
        return

    predictions = load_fase4_table("out_of_fold_predictions.csv")
    rows = predictions[predictions["image_name"] == stem]
    if rows.empty:
        return

    st.subheader("Predizione del modello (Fase 4)")
    validation = st.radio(
        "Validazione",
        sorted(rows["validation"].unique()),
        horizontal=True,
        key="oof_validation",
        help=(
            "A_casuale: split casuale, stima ottimistica perche' patch dello stesso "
            "caso possono finire sia in addestramento sia in test. B_blocchi: split "
            "a blocchi contigui, stima conservativa."
        ),
    )
    subset = rows[rows["validation"] == validation]
    models = sorted(subset["model"].unique())
    model = st.selectbox(
        "Modello", models,
        index=models.index("xgboost") if "xgboost" in models else 0,
        key="oof_model",
    )

    chosen = subset[subset["model"] == model]
    if chosen.empty:
        return
    _render_probability(float(chosen.iloc[0]["y_prob"]), truth=truth)
    st.caption(
        "Predizione **fuori-piega**: prodotta nella piega in cui questa patch era "
        "in test, da un modello che non l'aveva vista in addestramento. E' l'unica "
        "onesta per una patch del dataset."
    )


# ---------------------------------------------------------------------------
# Sezione 2 — analisi di un'immagine nuova
# ---------------------------------------------------------------------------
def render_analyzer(master: pd.DataFrame, significant: dict[str, bool]) -> None:
    st.warning(
        "**Non e' un dispositivo diagnostico.** Il modello e' stato addestrato su "
        "600 patch di due sole classi e raggiunge un AUC-ROC di 0.94 in validazione "
        "conservativa: e' uno strumento di ricerca, non un supporto alla decisione "
        "clinica. Su un'immagine che non sia una patch linfonodale H&E risponde "
        "comunque, con una sicurezza che non ha alcun fondamento — non avendo mai "
        "visto nulla di diverso, non puo' sapere di essere fuori dal proprio dominio.",
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

    if (FASE4_DIR / "best_model.joblib").exists():
        st.subheader("Predizione del modello")
        with st.spinner("Classificazione e spiegazione..."):
            explanation = explain_patch(get_classifier(), result["features"])
        _render_probability(explanation.probability)

        st.subheader("Spiegazione della decisione")
        st.caption(
            "Contributo di ciascun biomarcatore a QUESTA decisione: non quanto "
            "conta in generale, ma quanto ha pesato qui e in quale direzione."
        )
        _render_waterfall(explanation)
    else:
        st.info(
            "Modello della Fase 4 non presente: eseguire "
            "`python src/04_classification.py` per abilitare la predizione.",
            icon="ℹ️",
        )

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
# Sezione 3 — spiegabilita' e analisi dei risultati (Fase 4)
# ---------------------------------------------------------------------------
def _render_validation_gap(metrics: pd.DataFrame) -> None:
    """La forbice fra validazione ottimistica e conservativa."""
    st.subheader("Quanto del punteggio era leakage")
    st.caption(
        "Le 600 patch vengono da ~221 casi, con piu' patch per caso, e il dataset "
        "non contiene identificativi di paziente. Con uno split casuale patch dello "
        "stesso vetrino finiscono da entrambe le parti e il modello puo' "
        "riconoscere il vetrino invece della biologia. Ogni modello e' quindi "
        "valutato due volte: **la forbice fra le due misura il leakage invece di "
        "nasconderlo**."
    )

    summary = metrics.groupby(["model", "validation"])["auc_roc"].mean().unstack()
    summary["forbice"] = summary["A_casuale"] - summary["B_blocchi"]
    summary = summary.sort_values("B_blocchi", ascending=False)

    for model, row in summary.iterrows():
        left, middle, right = st.columns(3)
        left.metric(f"{model} — A casuale", f"{row['A_casuale']:.4f}",
                    help="Split casuale: stima ottimistica, contaminata dal leakage.")
        middle.metric("B a blocchi", f"{row['B_blocchi']:.4f}",
                      help="Split a blocchi contigui: stima conservativa. E' quella da citare.")
        right.metric("Forbice", f"{row['forbice']:+.4f}",
                     help="Quanto del punteggio spariva togliendo il leakage.")

    best = summary.index[0]
    st.success(
        f"**Il numero da portare in tesi e' {summary.loc[best, 'B_blocchi']:.4f}** "
        f"di AUC-ROC ({best}, validazione conservativa). La forbice e' di appena "
        f"{summary.loc[best, 'forbice']:.3f}: il leakage c'era, ma pesava poco.",
        icon="✅",
    )

    gap_figure = IMG_FASE4_DIR / "validation_gap.png"
    if gap_figure.exists():
        st.image(str(gap_figure), caption="Forbice fra le due validazioni, per modello")


def _render_global_explainability() -> None:
    """Classifica SHAP globale, con gli effetti non monotoni dichiarati come tali."""
    importance = load_fase4_table("shap_importance.csv")

    st.subheader("Quali biomarcatori decidono, e in che direzione")
    st.caption(
        "`importance` e' la media dei contributi in valore assoluto. `direction` "
        "dice verso quale classe spingono i valori alti — ma solo quando ha senso "
        "dirlo."
    )

    non_monotone = importance[importance["direction"] == "non monotona"]
    monotone = importance[importance["direction"] != "non monotona"]

    st.dataframe(importance, width="stretch", hide_index=True)

    if not non_monotone.empty:
        names = ", ".join(f"`{f}`" for f in non_monotone["feature"].head(5))
        with st.expander(
            f"⚠️ {len(non_monotone)} biomarcatori hanno un effetto NON monotono — "
            "perche' e' importante", expanded=True,
        ):
            st.markdown(
                f"""
Per {names} non esiste una direzione da dichiarare: **valori estremi in
entrambi i sensi spingono verso la stessa classe**.

Il caso da raccontare in discussione e' `solidity_mean`. Le medie delle due
classi sono quasi identiche — il test univariato della Fase 3 non lo trova
significativo — ma le **dispersioni** sono molto diverse. Il modello lo usa
comunque, e lo usa a U: nuclei con solidita' molto bassa *e* nuclei con
solidita' molto alta spingono entrambi verso il linfoma.

Una correlazione lineare fra valore e contributo qui restituirebbe un numero
vicino a zero, e riassumerla con una freccia direbbe una cosa falsa. Il profilo
per quintili nella colonna `quintile_profile` mostra la forma reale: si legge da
sinistra (valori bassi) a destra (valori alti).

**E' un risultato, non un difetto**: un biomarcatore che l'analisi univariata
scarta si rivela utile a un modello multivariato, e solo guardando la forma
dell'effetto si capisce perche'.
"""
            )

    st.caption(
        f"{len(monotone)} biomarcatori su {len(importance)} hanno un effetto "
        "monotono e una direzione dichiarabile."
    )

    comparison = IMG_FASE4_DIR / "shap_vs_univariate.png"
    if comparison.exists():
        st.image(
            str(comparison),
            caption="Gerarchia SHAP (multivariata) contro effect size univariati della Fase 3",
        )


def _render_local_explainability(master: pd.DataFrame) -> None:
    """Spiegazione locale calcolata dal vivo su una patch scelta dall'utente."""
    st.subheader("Spiegazione di un caso singolo")
    st.caption(
        "Scegli una patch e osserva **perche'** il modello decide cosi': quali "
        "biomarcatori spostano la decisione, di quanto e in quale direzione. "
        "Calcolato sul momento, non una figura preparata."
    )

    left, right = st.columns([1, 2])
    category = left.selectbox("Categoria", CATEGORIES, format_func=short_label,
                              key="xai_categoria")
    candidates = master.loc[master["category"] == category, "image_name"].tolist()
    stem = right.selectbox("Patch", candidates, key="xai_patch")

    row = master.loc[master["image_name"] == stem]
    if row.empty:
        return
    values = row.iloc[0]
    features = {k: float(v) for k, v in values.items() if k not in ("category", "image_name")}

    with st.spinner("Calcolo della spiegazione..."):
        explanation = explain_patch(get_classifier(), features)

    _render_probability(explanation.probability, truth=int(values["target"]))
    _render_waterfall(explanation)
    st.caption(
        "Nota: la spiegazione usa il modello finale, addestrato su tutte le 600 "
        "patch. Serve a mostrare il **ragionamento**, non a stimare le prestazioni "
        "— per quelle valgono solo le predizioni fuori-piega."
    )


def render_explainability(master: pd.DataFrame) -> None:
    if not METRICS_CSV.exists() or not (FASE4_DIR / "best_model.joblib").exists():
        st.error(
            "Risultati della Fase 4 mancanti. Eseguire prima "
            "`python src/04_classification.py`."
        )
        return

    metrics = load_fase4_table("metrics_by_model.csv")
    _render_validation_gap(metrics)
    st.divider()
    _render_global_explainability()
    st.divider()
    _render_local_explainability(master)
    st.divider()

    st.subheader("Robustezza e dettagli del metodo")
    sensitivity_tab, roc_tab, reduction_tab, folds_tab = st.tabs(
        ["Sensibilita' al blocco", "Curve ROC", "Riduzione 47 -> 33", "Metriche per piega"]
    )

    with sensitivity_tab:
        st.caption(
            "La conclusione non deve dipendere da un parametro arbitrario: la "
            "validazione conservativa e' ripetuta con blocchi di dimensione "
            "diversa. Un degrado al crescere del blocco indicherebbe una "
            "dipendenza reale dal vicinato."
        )
        if SENSITIVITY_CSV.exists():
            sensitivity = load_fase4_table("block_size_sensitivity.csv")
            pivot = sensitivity.groupby(["block_size", "model"])["auc_roc"].mean().unstack()
            st.line_chart(pivot)
            st.dataframe(pivot, width="stretch")

    with roc_tab:
        roc_figure = IMG_FASE4_DIR / "roc_curves.png"
        if roc_figure.exists():
            st.image(str(roc_figure), caption="ROC fuori-piega, entrambe le validazioni")

    with reduction_tab:
        st.caption(
            "Fra biomarcatori quasi identici (|rho| > 0.90) ne resta uno solo: "
            "SHAP dividerebbe il merito fra le copie, facendole apparire entrambe "
            "meno importanti di quanto sono. A parita' di ridondanza si tiene la "
            "variabile **piu' leggibile clinicamente**, non quella statisticamente "
            "piu' forte."
        )
        if REDUCTION_CSV.exists():
            reduction = load_fase4_table("feature_reduction.csv")
            st.dataframe(
                reduction[~reduction["kept"]][["feature", "representative"]].rename(
                    columns={"feature": "scartato", "representative": "rappresentato da"}
                ),
                width="stretch", hide_index=True,
            )

    with folds_tab:
        st.caption(
            "Le medie nascondono la variabilita': con poche unita' indipendenti "
            "nella validazione a blocchi, la dispersione fra pieghe e' ampia e va "
            "guardata prima di dichiarare un modello migliore di un altro."
        )
        st.dataframe(metrics, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Sezione 4 — risultati statistici della Fase 3
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

    explorer, analyzer, explainability, results = st.tabs(
        ["Esplora dataset", "Analizza immagine", "Spiegabilita'", "Risultati Fase 3"]
    )
    with explorer:
        render_explorer(master, significant)
    with analyzer:
        render_analyzer(master, significant)
    with explainability:
        render_explainability(master)
    with results:
        render_results(separability)


# Streamlit esegue lo script come "__main__": la guardia lascia il modulo
# importabile dai test senza far partire l'intera interfaccia.
if __name__ == "__main__":
    main()
