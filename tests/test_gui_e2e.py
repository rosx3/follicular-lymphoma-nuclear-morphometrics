"""
Test end-to-end dell'interfaccia, con un browser vero (Playwright).

PERCHE' ESISTONO. I test in test_gui.py girano con streamlit.testing.v1.AppTest,
che esegue lo script senza browser: veloci e precisi, ma **non sanno simulare il
caricamento di un file**. Restava scoperto proprio il percorso piu' visibile
dell'app — l'utente carica un'immagine, la pipeline gira dal vivo, il modello
classifica e spiega. Qui quel percorso viene fatto davvero: si avvia il server,
si apre Chromium, si carica un file dal disco e si guarda cosa compare.

COSTO. Sono lenti (avvio del server, fit del normalizzatore di Macenko,
segmentazione) e richiedono il browser scaricato da Playwright. Sono percio'
marcati `e2e` ed esclusi dalla suite veloce:

    python -m pytest tests/ -q -m "not e2e"     # suite rapida
    python -m pytest tests/test_gui_e2e.py -q   # solo end-to-end

Preparazione (una volta sola):

    pip install pytest-playwright
    python -m playwright install chromium
"""

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("playwright", reason="Playwright non installato: vedi docstring")
from playwright.sync_api import Locator, Page, expect  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
GUI_PATH = BASE_DIR / "src" / "gui.py"
SAMPLE_PATCH = BASE_DIR / "data" / "raw" / "follicular_lymphoma" / "FL_examples (1).jpg"

# L'avvio importa i moduli della pipeline e legge i CSV: alcuni secondi.
SERVER_STARTUP_TIMEOUT = 180
# Il fit di Macenko e la segmentazione, la prima volta, non sono istantanei.
PROCESSING_TIMEOUT_MS = 180_000

pytestmark = pytest.mark.e2e


def _free_port() -> int:
    """Una porta libera, per non collidere con un'istanza gia' aperta dall'utente."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def streamlit_server() -> str:
    """Avvia l'app su una porta libera e la spegne a fine modulo."""
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(GUI_PATH),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            output = process.stdout.read().decode("utf-8", "replace")
            pytest.fail(f"il server e' uscito durante l'avvio:\n{output}")
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    else:
        process.terminate()
        pytest.fail(f"il server non ha risposto entro {SERVER_STARTUP_TIMEOUT}s")

    yield url

    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.fixture
def analyzer(page: Page, streamlit_server: str) -> Locator:
    """
    Il pannello della sezione 'Analizza immagine'.

    Si restituisce il pannello e non la pagina perche' Streamlit tiene nel DOM il
    contenuto di TUTTE le tab, anche di quelle non attive: cercare un testo sulla
    pagina intera lo troverebbe anche nelle altre sezioni. Ogni pannello e'
    etichettato col nome della propria tab, il che permette di circoscrivere.
    """
    page.goto(streamlit_server, wait_until="domcontentloaded")
    tab = page.get_by_role("tab", name="Analizza immagine")
    expect(tab).to_be_visible(timeout=60_000)
    tab.click()
    return page.get_by_label("Analizza immagine")


def _upload(panel: Locator, path: Path) -> None:
    assert path.exists(), f"immagine di prova mancante: {path}"
    panel.locator('input[type="file"]').set_input_files(str(path))


def _upload_sample(panel: Locator) -> None:
    _upload(panel, SAMPLE_PATCH)


# --------------------------------------------------------------------------
# Il percorso che AppTest non raggiunge: caricare davvero un'immagine
# --------------------------------------------------------------------------
def test_uploading_an_image_runs_the_pipeline_and_shows_every_stage(analyzer: Locator):
    _upload_sample(analyzer)

    expect(analyzer.get_by_text("Nuclei segmentati")).to_be_visible(
        timeout=PROCESSING_TIMEOUT_MS
    )
    for caption in ("1. Caricata", "2. Macenko", "4. Contorni dei nuclei"):
        expect(analyzer.get_by_text(caption)).to_be_visible()


def test_uploading_an_image_produces_a_plausible_nucleus_count(analyzer: Locator):
    """
    La patch di prova ne ha 170 nel dataset. L'analisi dal vivo ricalcola la Fase
    1 in memoria invece di rileggere il PNG salvato, quindi lo scarto atteso e'
    di pochi punti percentuali: si controlla l'ordine di grandezza, non l'identita'.
    """
    _upload_sample(analyzer)

    metric = analyzer.locator('[data-testid="stMetric"]', has_text="Nuclei segmentati")
    expect(metric).to_be_visible(timeout=PROCESSING_TIMEOUT_MS)

    counted = int(metric.get_by_test_id("stMetricValue").inner_text().strip())
    assert 140 <= counted <= 200, f"nuclei segmentati fuori scala: {counted}"


def test_uploading_an_image_classifies_it_and_explains_the_decision(analyzer: Locator):
    """
    Il collegamento del modello alla GUI: predizione e spiegazione locale devono
    comparire davvero, non solo esistere come funzioni.
    """
    _upload_sample(analyzer)

    expect(analyzer.get_by_text("Probabilita' di linfoma follicolare")).to_be_visible(
        timeout=PROCESSING_TIMEOUT_MS
    )
    # Il titolo del waterfall e' disegnato DENTRO la figura matplotlib, quindi
    # non e' testo cercabile: si verificano l'intestazione della sezione, la
    # didascalia sull'additivita' e la presenza della figura stessa.
    expect(analyzer.get_by_text("Spiegazione della decisione")).to_be_visible()
    expect(analyzer.get_by_text("somma dei contributi")).to_be_visible()
    expect(analyzer.locator("img").last).to_be_visible()


def test_the_analyzer_warns_it_is_not_a_diagnostic_device(analyzer: Locator):
    expect(analyzer.get_by_text("Non e' un dispositivo diagnostico")).to_be_visible()


def test_a_non_histological_image_is_rejected_instead_of_being_classified(
    analyzer: Locator, tmp_path: Path
):
    """
    Un'immagine senza nuclei non deve produrre una predizione: senza nuclei non
    ci sono biomarcatori, e classificarla comunque darebbe un numero inventato.
    """
    import cv2
    import numpy as np

    blank = tmp_path / "bianco.png"
    cv2.imwrite(str(blank), np.full((224, 224, 3), 255, dtype=np.uint8))
    _upload(analyzer, blank)

    expect(analyzer.get_by_text("Nessun nucleo segmentato")).to_be_visible(
        timeout=PROCESSING_TIMEOUT_MS
    )
    expect(analyzer.get_by_text("Probabilita' di linfoma follicolare")).not_to_be_visible()
