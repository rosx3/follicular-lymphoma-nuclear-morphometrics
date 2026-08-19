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
