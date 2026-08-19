"""
Test della calibrazione spaziale (src/calibration.py).

Il valore era duplicato in quattro moduli e, in run_pipeline.py, comparivano
anche due letterali 0.23 non collegati alla costante. Una revisione della
calibrazione richiedeva quindi di trovare e correggere cinque punti diversi,
con il rischio concreto di lasciarne indietro uno e produrre CSV con colonne
calibrate in modo incoerente fra loro.

Questi test fissano il valore corrente, la coerenza delle grandezze derivate,
e il fatto che nessun altro modulo ridefinisca la scala per conto proprio.
"""

import re
from pathlib import Path

import pytest

from calibration import (
    MICRONS_PER_PIXEL,
    PATCH_AREA_UM2,
    PATCH_SIDE_UM,
    PATCH_SIZE_PX,
    PIXEL_AREA_UM2,
    px2_to_um2,
    px_to_um,
)

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def test_the_calibration_is_the_value_derived_from_the_source_paper():
    """Carreras et al. (2025) esportano le patch a 200x = obiettivo 20x.

    Vedi reports/fase3_report.md 3.4 per la verifica completa.
    """
    assert MICRONS_PER_PIXEL == 0.46


def test_pixel_area_is_the_square_of_the_pixel_side():
    assert PIXEL_AREA_UM2 == pytest.approx(MICRONS_PER_PIXEL**2)


def test_patch_side_and_area_follow_from_the_pixel_size():
    assert PATCH_SIDE_UM == pytest.approx(PATCH_SIZE_PX * MICRONS_PER_PIXEL)
    assert PATCH_AREA_UM2 == pytest.approx(PATCH_SIDE_UM**2)


def test_conversion_helpers_agree_with_the_constants():
    assert px_to_um(10.0) == pytest.approx(10.0 * MICRONS_PER_PIXEL)
    assert px2_to_um2(100.0) == pytest.approx(100.0 * PIXEL_AREA_UM2)


def test_a_length_converts_quadratically_when_it_becomes_an_area():
    """Controllo dimensionale: raddoppiare i px quadruplica l'area."""
    assert px2_to_um2(4.0) == pytest.approx(px_to_um(2.0) ** 2)


# Letterali che rappresentano una dimensione di pixel in micron e che non
# devono comparire fuori da calibration.py.
_FORBIDDEN_LITERALS = re.compile(r"\b0\.23\b|\b0\.0529\b|\b0\.46\b|\b0\.2116\b")


def _source_files_other_than_calibration():
    return [p for p in SRC_DIR.glob("*.py") if p.name != "calibration.py"]


@pytest.mark.parametrize(
    "source_file", _source_files_other_than_calibration(), ids=lambda p: p.name
)
def test_no_module_hardcodes_the_pixel_size(source_file):
    offenders = [
        f"{source_file.name}:{i}: {line.strip()}"
        for i, line in enumerate(source_file.read_text(encoding="utf-8").splitlines(), 1)
        if _FORBIDDEN_LITERALS.search(line)
    ]

    assert not offenders, (
        "la dimensione del pixel va importata da calibration.py, non riscritta:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "source_file", _source_files_other_than_calibration(), ids=lambda p: p.name
)
def test_no_module_redefines_the_calibration_constants(source_file):
    redefinitions = [
        f"{source_file.name}:{i}: {line.strip()}"
        for i, line in enumerate(source_file.read_text(encoding="utf-8").splitlines(), 1)
        if re.match(r"\s*(MICRONS_PER_PIXEL|PIXEL_AREA_UM2|PATCH_AREA_UM2)\s*=", line)
    ]

    assert not redefinitions, (
        "costante di calibrazione ridefinita fuori da calibration.py:\n"
        + "\n".join(redefinitions)
    )
