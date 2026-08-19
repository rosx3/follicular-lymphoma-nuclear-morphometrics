"""
Test delle convenzioni di naming della pipeline (src/naming.py).

Contesto (problema B1/B2 dell'audit Fase 3): i file su disco prodotti dalle
Fasi 1 e 2 usano i suffissi _norm.png / _hchannel.png / _mask.png, mentre
run_pipeline.py ne ricostruiva i nomi con regole diverse e incoerenti, e la
categoria era rappresentata con tre convenzioni diverse
("FL", "Follicular Lymphoma", "follicular_lymphoma").
"""

import pytest

from naming import (
    CATEGORY_FL,
    CATEGORY_REACTIVE,
    h_channel_name,
    mask_name,
    normalize_category,
    overlay_name,
    rgb_normalized_name,
    stem_from_h_channel_name,
    stem_from_mask_name,
    target_from_category,
)

# Stem realmente presente nel dataset, usato come ancora contro i file su disco.
SAMPLE_STEM = "FL_examples (1)"


def test_rgb_normalized_name_matches_file_on_disk(fase1_dir):
    path = fase1_dir / CATEGORY_FL / "rgb_normalized" / rgb_normalized_name(SAMPLE_STEM)
    assert path.exists(), f"Nome RGB normalizzata non corrispondente al disco: {path}"


def test_h_channel_name_matches_file_on_disk(fase1_dir):
    path = fase1_dir / CATEGORY_FL / "h_channel" / h_channel_name(SAMPLE_STEM)
    assert path.exists(), f"Nome H-channel non corrispondente al disco: {path}"


def test_mask_name_matches_file_on_disk(fase2_dir):
    path = fase2_dir / CATEGORY_FL / "masks" / mask_name(SAMPLE_STEM)
    assert path.exists(), f"Nome maschera non corrispondente al disco: {path}"


def test_overlay_name_matches_file_on_disk(fase2_dir):
    path = fase2_dir / CATEGORY_FL / "overlays" / overlay_name(SAMPLE_STEM)
    assert path.exists(), f"Nome overlay non corrispondente al disco: {path}"


def test_stem_from_mask_name_is_inverse_of_mask_name():
    assert stem_from_mask_name(mask_name(SAMPLE_STEM)) == SAMPLE_STEM


def test_stem_from_mask_name_accepts_a_full_path():
    assert stem_from_mask_name("data/fase2/masks/" + mask_name(SAMPLE_STEM)) == SAMPLE_STEM


def test_stem_from_mask_name_rejects_a_name_without_the_mask_suffix():
    with pytest.raises(ValueError):
        stem_from_mask_name("FL_examples (1).png")


def test_stem_from_h_channel_name_is_inverse_of_h_channel_name():
    """La Fase 2 itera gli H-channel: deve ricavarne lo stem senza il suffisso."""
    assert stem_from_h_channel_name(h_channel_name(SAMPLE_STEM)) == SAMPLE_STEM


def test_stem_from_h_channel_name_rejects_a_name_without_the_suffix():
    with pytest.raises(ValueError):
        stem_from_h_channel_name("FL_examples (1).png")


@pytest.mark.parametrize(
    "raw",
    ["FL", "fl", "Follicular Lymphoma", "follicular_lymphoma", " Follicular  Lymphoma "],
)
def test_normalize_category_maps_every_fl_variant_to_canonical(raw):
    assert normalize_category(raw) == CATEGORY_FL


@pytest.mark.parametrize(
    "raw",
    ["REACTIVE", "reactive", "Reactive Tissue", "reactive_tissue", " Reactive  Tissue "],
)
def test_normalize_category_maps_every_reactive_variant_to_canonical(raw):
    assert normalize_category(raw) == CATEGORY_REACTIVE


def test_normalize_category_rejects_unknown_label():
    with pytest.raises(ValueError):
        normalize_category("linfonodo_ignoto")


def test_target_is_1_for_follicular_lymphoma_and_0_for_reactive():
    assert target_from_category(CATEGORY_FL) == 1
    assert target_from_category(CATEGORY_REACTIVE) == 0


def test_target_accepts_non_canonical_labels():
    assert target_from_category("Follicular Lymphoma") == 1
    assert target_from_category("REACTIVE") == 0
