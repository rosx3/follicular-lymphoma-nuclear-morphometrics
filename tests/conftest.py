"""
Configurazione pytest condivisa.

Rende importabili i moduli in src/ (che non e' un package) e mette a
disposizione i percorsi delle directory di dati prodotte dalle Fasi 1 e 2.
"""

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def base_dir() -> Path:
    return BASE_DIR


@pytest.fixture(scope="session")
def fase1_dir(base_dir: Path) -> Path:
    return base_dir / "data" / "fase1_preprocessing"


@pytest.fixture(scope="session")
def fase2_dir(base_dir: Path) -> Path:
    return base_dir / "data" / "fase2_segmentation"


def pytest_configure(config):
    """
    Marker per i test end-to-end col browser (tests/test_gui_e2e.py).

    Sono lenti e richiedono Chromium scaricato da Playwright, quindi non fanno
    parte della suite rapida: `python -m pytest tests/ -q -m "not e2e"`.
    """
    config.addinivalue_line(
        "markers", "e2e: test end-to-end con un browser vero (lenti, richiedono Playwright)"
    )
