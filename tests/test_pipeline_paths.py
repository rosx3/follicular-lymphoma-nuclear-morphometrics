"""
Test di risoluzione dei percorsi della Fase 3 sull'intero dataset.

Contesto (problema B1 dell'audit Fase 3): run_fase3 ricostruiva il percorso
della RGB normalizzata con mask_p.name.replace("_mask.png", ".png"), che non
corrisponde ad alcun file su disco (i file reali terminano con "_norm.png").
L'effetto era che l'anteprima grafica veniva saltata in silenzio, senza errore.
Inoltre l'H-channel — input necessario alle feature di tessitura — non veniva
risolto affatto.
"""

import pytest

from naming import (
    CATEGORIES,
    CATEGORY_FL,
    h_channel_name,
    iter_h_channel_inputs,
    iter_patch_inputs,
    mask_name,
    rgb_normalized_name,
)


@pytest.fixture
def fake_dataset(tmp_path):
    """Mini-dataset su disco con una sola patch completa e coerente."""
    stem = "FL_examples (1)"
    mask_dir = tmp_path / "fase2" / CATEGORY_FL / "masks"
    rgb_dir = tmp_path / "fase1" / CATEGORY_FL / "rgb_normalized"
    h_dir = tmp_path / "fase1" / CATEGORY_FL / "h_channel"
    for d in (mask_dir, rgb_dir, h_dir):
        d.mkdir(parents=True)
    (mask_dir / mask_name(stem)).write_bytes(b"")
    (rgb_dir / rgb_normalized_name(stem)).write_bytes(b"")
    (h_dir / h_channel_name(stem)).write_bytes(b"")
    return tmp_path / "fase1", tmp_path / "fase2", stem


def test_iter_patch_inputs_yields_the_three_inputs_of_each_patch(fake_dataset):
    fase1, fase2, stem = fake_dataset

    inputs = list(iter_patch_inputs(CATEGORY_FL, fase1, fase2))

    assert len(inputs) == 1
    patch = inputs[0]
    assert patch.stem == stem
    assert patch.category == CATEGORY_FL
    assert patch.mask_path.name == mask_name(stem)
    assert patch.rgb_path.name == rgb_normalized_name(stem)
    assert patch.h_channel_path.name == h_channel_name(stem)


def test_iter_patch_inputs_raises_instead_of_skipping_a_missing_rgb(fake_dataset):
    """Regressione B1: l'input mancante deve fallire rumorosamente, non in silenzio."""
    fase1, fase2, stem = fake_dataset
    (fase1 / CATEGORY_FL / "rgb_normalized" / rgb_normalized_name(stem)).unlink()

    with pytest.raises(FileNotFoundError, match=rgb_normalized_name(stem).replace("(", r"\(").replace(")", r"\)")):
        list(iter_patch_inputs(CATEGORY_FL, fase1, fase2))


def test_iter_patch_inputs_raises_instead_of_skipping_a_missing_h_channel(fake_dataset):
    fase1, fase2, stem = fake_dataset
    (fase1 / CATEGORY_FL / "h_channel" / h_channel_name(stem)).unlink()

    with pytest.raises(FileNotFoundError):
        list(iter_patch_inputs(CATEGORY_FL, fase1, fase2))


def test_iter_patch_inputs_resolves_the_whole_real_dataset(fase1_dir, fase2_dir):
    """600 patch reali: ogni maschera deve trovare RGB e H-channel esistenti."""
    total = sum(len(list(iter_patch_inputs(cat, fase1_dir, fase2_dir))) for cat in CATEGORIES)
    assert total == 600


def test_iter_h_channel_inputs_yields_h_channel_and_rgb_of_each_image(fake_dataset):
    fase1, _fase2, stem = fake_dataset

    inputs = list(iter_h_channel_inputs(CATEGORY_FL, fase1))

    assert len(inputs) == 1
    image = inputs[0]
    assert image.stem == stem
    assert image.category == CATEGORY_FL
    assert image.h_channel_path.name == h_channel_name(stem)
    assert image.rgb_path.name == rgb_normalized_name(stem)


def test_iter_h_channel_inputs_raises_instead_of_skipping_a_missing_rgb(fake_dataset):
    """Regressione: in Fase 2 l'overlay veniva saltato in silenzio se mancava la RGB."""
    fase1, _fase2, stem = fake_dataset
    (fase1 / CATEGORY_FL / "rgb_normalized" / rgb_normalized_name(stem)).unlink()

    with pytest.raises(FileNotFoundError):
        list(iter_h_channel_inputs(CATEGORY_FL, fase1))


def test_iter_h_channel_inputs_covers_the_whole_real_dataset(fase1_dir):
    total = sum(len(list(iter_h_channel_inputs(cat, fase1_dir))) for cat in CATEGORIES)
    assert total == 600


def test_iter_patch_inputs_covers_every_mask_of_the_real_dataset(fase1_dir, fase2_dir):
    """Nessuna maschera deve restare fuori dall'iterazione."""
    for category in CATEGORIES:
        on_disk = {p.name for p in (fase2_dir / category / "masks").glob("*.png")}
        iterated = {p.mask_path.name for p in iter_patch_inputs(category, fase1_dir, fase2_dir)}
        assert iterated == on_disk
