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


def test_fase3_csv_has_exactly_the_50_contracted_columns(pipeline):
    """Il CSV per patch e' l'input della Fase 4: la sua forma e' un contratto."""
    import csv
    import importlib.util

    module, _stems = pipeline
    spec = importlib.util.spec_from_file_location(
        "mod_features_e2e", Path(__file__).resolve().parent.parent / "src" / "03_feature_extraction.py"
    )
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)

    with open(module.FASE3_DIR / "features_patches_master.csv", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    expected = list(features.PATCH_METADATA_COLUMNS) + list(features.PATCH_FEATURE_COLUMNS)
    assert header == expected, "il CSV non rispetta il contratto delle colonne"
    assert len(header) == 50


def test_fase3_populates_the_knn_and_texture_columns(pipeline):
    """Regressione: k-NN e tessitura devono essere davvero cablate in run_fase3."""
    import csv

    module, _stems = pipeline
    with open(module.FASE3_DIR / "features_patches_master.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for column in ("knn1_dist_mean_um", "knn3_dist_mean_um", "glcm_contrast", "lbp_entropy"):
        values = [r[column] for r in rows]
        assert all(v != "" for v in values), f"colonna {column} vuota"
        assert any(v not in ("", "nan", "0.0") for v in values), f"colonna {column} mai valorizzata"


def test_fase3_produces_one_row_per_patch_with_the_right_target(pipeline):
    import csv

    module, stems = pipeline
    expected = sum(len(s) for s in stems.values())

    with open(module.FASE3_DIR / "features_patches_master.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == expected, "la Fase 3 ha saltato delle patch"
    for row in rows:
        assert row["target"] == ("1" if row["category"] == CATEGORY_FL else "0")


def test_fase3_writes_a_reproducible_metadata_file(pipeline):
    """I parametri che determinano i valori estratti devono essere ricostruibili.

    Senza questo file, i CSV della Fase 3 non sono riproducibili: quantizzazione
    GLCM, raggio LBP, k delle distanze e calibrazione spaziale non sarebbero
    desumibili dai dati.
    """
    import json

    module, stems = pipeline
    metadata_path = module.FASE3_DIR / "feature_extraction_metadata.json"

    assert metadata_path.exists(), "metadata di riproducibilita' non generato"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["conteggi"]["patch_processate"] == sum(len(s) for s in stems.values())
    assert metadata["conteggi"]["patch_in_errore"] == 0
    assert metadata["feature"]["n_feature"] == 47
    assert metadata["feature"]["n_metadati"] == 3

    assert metadata["calibrazione"]["microns_per_pixel"] == 0.46
    assert metadata["parametri_glcm"]["levels"] == 64
    assert metadata["parametri_glcm"]["angles_deg"] == [0, 45, 90, 135]
    assert metadata["parametri_glcm"]["mascherato_sui_nuclei"] is True
    assert metadata["parametri_lbp"]["method"] == "uniform"
    assert metadata["parametri_knn"]["k"] == [1, 3]


def test_the_metadata_lists_the_same_columns_the_csv_contains(pipeline):
    """Il file di metadati non deve poter divergere dal CSV che descrive."""
    import csv
    import json

    module, _stems = pipeline
    metadata = json.loads(
        (module.FASE3_DIR / "feature_extraction_metadata.json").read_text(encoding="utf-8")
    )
    with open(module.FASE3_DIR / "features_patches_master.csv", newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    declared = metadata["feature"]["colonne_metadati"] + metadata["feature"]["colonne_feature"]
    assert declared == header


def test_the_metadata_records_the_library_versions_used(pipeline):
    """Le versioni contano: skimage cambia i default di graycomatrix fra release."""
    import json

    module, _stems = pipeline
    ambiente = json.loads(
        (module.FASE3_DIR / "feature_extraction_metadata.json").read_text(encoding="utf-8")
    )["ambiente"]

    for library in ("python", "numpy", "scipy", "scikit-image"):
        assert ambiente.get(library), f"versione mancante per {library}"


def test_fase3_generates_the_morphometry_preview(pipeline):
    """L'anteprima veniva saltata in silenzio prima del fix del naming."""
    module, _stems = pipeline
    preview = module.IMG_FASE3_DIR / "morphometry_regions_preview.png"

    assert preview.exists(), "anteprima citomorfometrica non generata"
    assert preview.stat().st_size > 10_000, "anteprima generata ma vuota"
