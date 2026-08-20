"""
Guardie sullo script di benchmark eseguito fuori dal repository.

MOTIVAZIONE. Il benchmark della Fase 2 (GT Cellpose indipendente + U-Net) gira
su Google Colab, non in locale: qui non si puo' eseguire, perche' richiede
cellpose e una GPU. Ma proprio perche' vive fuori dalla pipeline testata, e' il
punto in cui il progetto ha gia' pagato due volte:

  1. Le 600 maschere della Fase 2 furono prodotte da un runner esterno che
     chiamava la segmentazione coi default impliciti del modulo. Quando i
     default cambiarono, il dataset divento' irriproducibile senza che nulla
     lo segnalasse (vedi test_segmentation_reproducibility.py).
  2. Il benchmark Cellpose ha ripetuto lo stesso errore: chiamava
     `segment_nuclei_watershed(h_img)` senza parametri, e ha quindi misurato
     una configurazione (min_distance=12, min_area_px=30) diversa da quella che
     ha prodotto il dataset — verificato sui conteggi `ws_n_pred` registrati,
     10 patch su 10.

  Inoltre la Ground Truth di Cellpose veniva tenuta in memoria e mai scritta su
  disco: alla chiusura della sessione Colab e' andata persa, e le metriche
  pubblicate non sono piu' verificabili ne' ricalcolabili.

Questi test non eseguono lo script: ne ispezionano il sorgente, nello stesso
spirito delle guardie di test_calibration.py. Servono a impedire che il
benchmark torni a dipendere da default impliciti o a buttare via la GT.
"""

import re
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = BASE_DIR / "scratch" / "run_colab_benchmark.py"


@pytest.fixture(scope="module")
def source() -> str:
    if not SCRIPT_PATH.exists():
        pytest.skip(f"{SCRIPT_PATH} non presente")
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_every_segmentation_call_states_its_parameters(source: str):
    """
    Nessuna chiamata alla segmentazione puo' affidarsi ai default del modulo.

    E' l'errore che ha reso il benchmark non rappresentativo del dataset: se un
    domani i default cambiassero di nuovo, uno script che non li dichiara
    cambierebbe silenziosamente cio' che misura.
    """
    # Vanno bene sia i parametri per nome sia l'espansione di un dizionario di
    # configurazione: cio' che non va bene e' la chiamata nuda, che eredita i
    # default del modulo.
    explicit = ("min_distance=", "**WS_PARAMS")
    implicit = [
        f"riga {i}: {line.strip()}"
        for i, line in enumerate(source.splitlines(), 1)
        if re.search(r"segment_nuclei_watershed\(", line)
        and not any(marker in line for marker in explicit)
    ]

    assert not implicit, (
        "chiamate alla segmentazione senza parametri espliciti:\n" + "\n".join(implicit)
    )


def test_the_cellpose_ground_truth_is_written_to_disk(source: str):
    """
    La GT deve sopravvivere alla sessione Colab.

    Senza le maschere salvate, le metriche pubblicate nella tesi non sono
    verificabili: e' esattamente cio' che e' successo al run originale.
    """
    assert "cellpose_gt" in source, "lo script non genera piu' una GT Cellpose"
    assert re.search(r"imwrite\([^)]*gt", source, re.IGNORECASE), (
        "la GT Cellpose non viene scritta su disco: alla chiusura della sessione "
        "andrebbe persa, come nel run originale"
    )


def test_the_run_records_its_own_provenance(source: str):
    """
    Versione di cellpose e parametri usati vanno registrati insieme ai risultati.

    I metadati del run originale annotavano solo "Cellpose v4.x": non abbastanza
    per rigenerare la stessa Ground Truth.
    """
    assert "cellpose.version" in source or "cellpose_version" in source, (
        "lo script non registra la versione di cellpose con cui e' stata generata la GT"
    )
    assert "benchmark_metadata" in source, (
        "lo script non scrive un file di metadati con i parametri del run"
    )
