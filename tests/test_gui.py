"""
Test dell'interfaccia Streamlit (src/gui.py).

L'interfaccia e' un guscio di widget: la logica che produce i numeri sta in
src/gui_core.py ed e' gia' coperta da test_gui_core.py. Qui si verifica cio'
che quei test non possono vedere, cioe' che il guscio regga: che l'app parta
senza eccezioni, che le tre sezioni esistano, che la tabella dei biomarcatori
segua davvero la patch selezionata e che l'app dichiari di non fornire una
diagnosi (il classificatore e' la Fase 4, ancora non implementata).

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


def test_the_app_offers_the_three_sections(app: AppTest):
    assert [tab.label for tab in app.tabs] == [
        "Esplora dataset",
        "Analizza immagine",
        "Risultati Fase 3",
    ]


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
def test_the_analyzer_states_that_it_does_not_provide_a_diagnosis(app: AppTest):
    disclaimers = [w.value for w in app.warning]

    assert any("Fase 4" in text for text in disclaimers), (
        "l'app non dichiara che la classificazione non e' implementata"
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
