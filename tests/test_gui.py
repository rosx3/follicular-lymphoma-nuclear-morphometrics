"""
Test dell'interfaccia Streamlit (src/gui.py).

L'interfaccia e' un guscio di widget: la logica che produce i numeri sta in
src/gui_core.py ed e' gia' coperta da test_gui_core.py. Qui si verifica cio'
che quei test non possono vedere, cioe' che il guscio regga: che l'app parta
senza eccezioni, che le quattro sezioni esistano, che la tabella dei biomarcatori
segua davvero la patch selezionata, e che la sezione di spiegabilita' pubblichi
la stima conservativa segnalando gli effetti non monotoni.

Il tutto gira headless con streamlit.testing.v1.AppTest: nessun browser.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parent.parent
GUI_PATH = BASE_DIR / "src" / "gui.py"

# L'avvio importa i moduli della pipeline e legge i CSV della Fase 3: il
# timeout predefinito di 3 secondi di AppTest non basta.
STARTUP_TIMEOUT = 180


@pytest.fixture
def app() -> AppTest:
    """App avviata da zero, cosi' ogni test parte da uno stato pulito."""
    return AppTest.from_file(str(GUI_PATH), default_timeout=STARTUP_TIMEOUT).run()


def _table_with_column(app: AppTest, column: str):
    """Tabella mostrata dall'app che contiene la colonna indicata."""
    for element in app.dataframe:
        if column in element.value.columns:
            return element.value
    raise AssertionError(f"nessuna tabella mostrata contiene la colonna '{column}'")


# --------------------------------------------------------------------------
# Avvio
# --------------------------------------------------------------------------
def test_the_app_starts_without_raising(app: AppTest):
    assert not app.exception, [e.value for e in app.exception]


def test_the_app_offers_the_four_sections(app: AppTest):
    # La sezione di spiegabilita' contiene a sua volta delle tab, che AppTest
    # elenca insieme a quelle principali: si verifica la sottosequenza.
    main_sections = [
        "Esplora dataset",
        "Analizza immagine",
        "Spiegabilita'",
        "Risultati Fase 3",
    ]
    labels = [tab.label for tab in app.tabs]

    assert [label for label in labels if label in main_sections] == main_sections


# --------------------------------------------------------------------------
# Sezione 1 — esplorazione di una patch gia' elaborata
# --------------------------------------------------------------------------
def test_the_explorer_shows_every_biomarker_of_the_selected_patch(app: AppTest):
    table = _table_with_column(app, "percentile_classe")

    assert len(table) == 47, "la tabella non mostra tutti i 47 biomarcatori"


def test_the_explorer_places_each_biomarker_inside_its_class_distribution(app: AppTest):
    table = _table_with_column(app, "percentile_classe")

    percentiles = table["percentile_classe"].dropna()
    assert len(percentiles) == 47
    assert percentiles.between(0.0, 100.0).all(), "percentili fuori da [0, 100]"


def test_choosing_another_patch_changes_the_biomarkers_shown(app: AppTest):
    first = _table_with_column(app, "percentile_classe")["valore"].tolist()

    patch_selector = app.selectbox(key="patch")
    patch_selector.select(patch_selector.options[1]).run()
    second = _table_with_column(app, "percentile_classe")["valore"].tolist()

    assert first != second, "la tabella non segue la patch selezionata"


def test_switching_category_offers_the_patches_of_that_category(app: AppTest):
    app.selectbox(key="categoria").select("reactive_tissue").run()

    assert not app.exception, [e.value for e in app.exception]
    assert all(stem.startswith("REACTIVE") for stem in app.selectbox(key="patch").options)


# --------------------------------------------------------------------------
# Sezione 2 — analisi di un'immagine nuova
# --------------------------------------------------------------------------
def test_the_analyzer_states_that_it_is_not_a_diagnostic_device(app: AppTest):
    """
    Ora che il modello e' collegato, l'avviso non riguarda piu' una funzione
    mancante ma il limite di cio' che c'e': uno strumento di ricerca addestrato
    su 600 patch di due sole classi, che risponde comunque anche fuori dominio.
    """
    disclaimers = [w.value for w in app.warning]

    assert any("dispositivo diagnostico" in text for text in disclaimers), (
        "l'app non dichiara di non essere un dispositivo diagnostico"
    )


def test_the_analyzer_declares_the_residual_gap_with_the_dataset(app: AppTest):
    """
    L'analisi dal vivo ricalcola la Fase 1 in memoria invece di rileggere i PNG
    salvati, quindi non coincide al 100% col dataset (scarto misurato su 16
    patch: mediana 0.0%, estremi -1.9% / +1.8%). Piccolo, ma va dichiarato:
    l'utente sta confrontando i propri numeri con distribuzioni di riferimento.
    """
    notes = [element.value for element in app.info]

    assert any("Fase 1" in text for text in notes), (
        "l'app non dichiara che l'analisi dal vivo ricalcola la Fase 1"
    )


def test_the_analyzer_positions_a_value_against_both_classes():
    """
    Il ramo a due classi della tabella, che AppTest non raggiunge: senza upload
    non viene mai renderizzato, ma e' quello che l'utente vede su un'immagine
    nuova, di cui la classe non e' nota.
    """
    import pandas as pd

    import gui

    master = gui.load_master_table.__wrapped__()
    patch = master.loc[master["image_name"] == "FL_examples (1)"].iloc[0]

    table = gui._positioned_table(
        patch,
        master,
        {"FL": "follicular_lymphoma", "REACTIVE": "reactive_tissue"},
        {},
    ).set_index("biomarcatore")

    assert set(table.columns) == {
        "valore",
        "media_FL",
        "percentile_FL",
        "media_REACTIVE",
        "percentile_REACTIVE",
        "significativo",
    }
    assert len(table) == 47
    # La patch appartiene a FL: la media della sua classe deve coincidere con
    # quella calcolata dal CSV, non con quella dell'altra classe.
    fl_only = master.loc[master["category"] == "follicular_lymphoma", "lbp_entropy"]
    assert table.loc["lbp_entropy", "media_FL"] == pytest.approx(fl_only.mean())
    assert not pd.isna(table.loc["lbp_entropy", "percentile_REACTIVE"])


# --------------------------------------------------------------------------
# Sezione 3 — risultati statistici della Fase 3
# --------------------------------------------------------------------------
def test_the_results_section_shows_the_separability_of_every_biomarker(app: AppTest):
    table = _table_with_column(app, "p_fdr")

    assert len(table) == 47
    assert table["significant"].sum() == 37, "i significativi non sono i 37 del report"


# --------------------------------------------------------------------------
# Sezione 3 — spiegabilita' (Fase 4)
# --------------------------------------------------------------------------
def test_the_explainability_section_publishes_the_conservative_estimate(app: AppTest):
    """
    Il numero da citare in tesi e' quello della validazione conservativa, non
    quello ottimistico: la sezione deve dirlo esplicitamente.
    """
    messages = [element.value for element in app.success]

    assert any("conservativa" in text for text in messages), (
        "la sezione non dichiara quale delle due stime va citata"
    )


def test_the_explainability_section_shows_the_shap_ranking(app: AppTest):
    table = _table_with_column(app, "direction")

    assert "importance" in table.columns
    assert len(table) == 33, "la classifica non copre i 33 biomarcatori del modello"


def test_the_explainability_section_flags_non_monotone_effects(app: AppTest):
    """
    Dichiarare una direzione per un effetto a U metterebbe un'affermazione falsa
    nella tesi: quei biomarcatori vanno segnalati, non riassunti con una freccia.
    """
    table = _table_with_column(app, "direction")

    assert "non monotona" in set(table["direction"]), (
        "nessun effetto segnalato come non monotono: la colonna direction non "
        "sta distinguendo i casi"
    )


def test_the_app_forces_a_headless_matplotlib_backend():
    """
    Streamlit esegue lo script in un thread separato. Col backend Tk (predefinito
    su Windows) la figura del waterfall nasce in quel thread e viene distrutta dal
    principale: Tk aborta il processo con "Tcl_AsyncDelete: async handler deleted
    by the wrong thread". Non e' un test che fallisce, e' la suite che muore —
    quindi la guardia sta qui, non nel messaggio d'errore di qualcun altro.
    """
    import matplotlib

    import gui  # noqa: F401  (l'import e' cio' che deve impostare il backend)

    assert matplotlib.get_backend().lower() == "agg"
