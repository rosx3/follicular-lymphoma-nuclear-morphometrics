"""
Test di integrazione della catena Fase 1 -> Fase 2 -> Fase 3.

Le tre fasi si scambiano dati esclusivamente attraverso i nomi dei file: la
Fase 1 li scrive, la Fase 2 li rilegge e ne scrive di nuovi, la Fase 3 li
rilegge tutti. Un disallineamento fra scrittura e rilettura non produce un
errore ma un risultato silenziosamente incompleto — ed e' esattamente quello
che era successo (audit Fase 3, problema B1).

Questo test esegue le tre fasi su un mini-dataset di 4 immagini reali copiate
in una directory temporanea, verificando che ogni fase trovi davvero cio' che
la precedente ha prodotto. Non tocca i dati in data/.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

from naming import (
    CATEGORIES,
    CATEGORY_FL,
    CATEGORY_REACTIVE,
    h_channel_name,
    mask_name,
    overlay_name,
    rgb_normalized_name,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RAW_DIR = BASE_DIR / "data" / "raw"

IMAGES_PER_CATEGORY = 2


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """run_pipeline con le directory di I/O reindirizzate su un dataset temporaneo."""
    tmp = tmp_path_factory.mktemp("pipeline_e2e")

    stems = {}
    for category in CATEGORIES:
        source_dir = RAW_DIR / category
        if not source_dir.is_dir():
            pytest.skip(f"Dataset grezzo non disponibile: {source_dir}")
        images = sorted(source_dir.glob("*.jpg"))[:IMAGES_PER_CATEGORY]
        if len(images) < IMAGES_PER_CATEGORY:
            pytest.skip(f"Meno di {IMAGES_PER_CATEGORY} immagini in {source_dir}")

        dest_dir = tmp / "raw" / category
        dest_dir.mkdir(parents=True)
        for image in images:
            shutil.copy2(image, dest_dir / image.name)
        stems[category] = [image.stem for image in images]

    spec = importlib.util.spec_from_file_location("mod_run_pipeline", SRC_DIR / "run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.RAW_DIR = tmp / "raw"
    module.FASE1_DIR = tmp / "fase1_preprocessing"
    module.FASE2_DIR = tmp / "fase2_segmentation"
    module.FASE3_DIR = tmp / "fase3_features"
    module.IMG_FASE3_DIR = tmp / "img_fase3"

    module.run_fase1(verbose=False)
    module.run_fase2(verbose=False)
    module.run_fase3(verbose=False)

    return module, stems


def test_fase1_writes_the_names_fase2_expects(pipeline):
    module, stems = pipeline
    for category, category_stems in stems.items():
        for stem in category_stems:
            assert (module.FASE1_DIR / category / "h_channel" / h_channel_name(stem)).exists()
            assert (module.FASE1_DIR / category / "rgb_normalized" / rgb_normalized_name(stem)).exists()


def test_fase2_writes_a_mask_and_an_overlay_for_every_image(pipeline):
    module, stems = pipeline
    for category, category_stems in stems.items():
        for stem in category_stems:
            assert (module.FASE2_DIR / category / "masks" / mask_name(stem)).exists()
            assert (module.FASE2_DIR / category / "overlays" / overlay_name(stem)).exists()


def test_fase2_centroids_use_the_canonical_category(pipeline):
    import csv

    module, _stems = pipeline
    with open(module.FASE2_DIR / "centroids_all.csv", newline="", encoding="utf-8") as f:
        categories = {row["category"] for row in csv.DictReader(f)}

    assert categories <= {CATEGORY_FL, CATEGORY_REACTIVE}, f"categorie non canoniche: {categories}"


def test_fase3_produces_one_row_per_patch_with_the_right_target(pipeline):
    import csv

    module, stems = pipeline
    expected = sum(len(s) for s in stems.values())

    with open(module.FASE3_DIR / "features_patches_master.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == expected, "la Fase 3 ha saltato delle patch"
    for row in rows:
        assert row["target"] == ("1" if row["category"] == CATEGORY_FL else "0")


def test_fase3_generates_the_morphometry_preview(pipeline):
    """L'anteprima veniva saltata in silenzio prima del fix del naming."""
    module, _stems = pipeline
    preview = module.IMG_FASE3_DIR / "morphometry_regions_preview.png"

    assert preview.exists(), "anteprima citomorfometrica non generata"
    assert preview.stat().st_size > 10_000, "anteprima generata ma vuota"
