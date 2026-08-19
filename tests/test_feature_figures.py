"""
Test delle figure di Fase 3 (Task 7 del piano).

Una figura non si valida con un assert sul suo aspetto, ma tre cose vanno
garantite comunque: che venga prodotta senza errori sui dati reali, che non sia
un'immagine vuota, e che i pannelli corrispondano alle feature richieste — un
grafico che mostra silenziosamente meno serie di quelle attese e un errore che
in un report passa inosservato.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_analysis import (
    plot_correlation_heatmap,
    plot_knn_distributions,
    plot_top_features_boxplots,
    separability_tests,
)

FL = "follicular_lymphoma"
REACTIVE = "reactive_tissue"

MIN_PNG_BYTES = 20_000  # una figura a 300 dpi non informativa pesa molto meno


@pytest.fixture
def synthetic_frame():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "image_name": [f"p{i}" for i in range(2 * n)],
        "category": [FL] * n + [REACTIVE] * n,
        "target": [1] * n + [0] * n,
        "lbp_entropy": np.concatenate([rng.normal(2.8, 0.2, n), rng.normal(3.0, 0.2, n)]),
        "knn1_dist_mean_um": np.concatenate([rng.normal(6.1, 0.3, n), rng.normal(5.9, 0.3, n)]),
        "knn3_dist_mean_um": np.concatenate([rng.normal(7.5, 0.3, n), rng.normal(7.2, 0.3, n)]),
        "knn1_dist_std_um": np.concatenate([rng.normal(1.2, 0.1, n), rng.normal(1.1, 0.1, n)]),
        "knn3_dist_std_um": np.concatenate([rng.normal(1.5, 0.1, n), rng.normal(1.4, 0.1, n)]),
        "area_um2_mean": np.concatenate([rng.normal(21.4, 1.0, n), rng.normal(23.0, 1.0, n)]),
        "glcm_homogeneity": np.concatenate([rng.normal(0.26, 0.02, n), rng.normal(0.24, 0.02, n)]),
    })


def test_the_boxplot_figure_has_one_panel_per_requested_feature(synthetic_frame, tmp_path):
    results = separability_tests(synthetic_frame)
    output = tmp_path / "boxplot.png"

    figure = plot_top_features_boxplots(synthetic_frame, results, output, n_features=4)

    assert len(figure.axes) == 4, "numero di pannelli diverso da quello richiesto"
    assert output.exists()
    assert output.stat().st_size > MIN_PNG_BYTES


def test_the_boxplot_shows_the_most_significant_features_first(synthetic_frame, tmp_path):
    results = separability_tests(synthetic_frame)

    figure = plot_top_features_boxplots(synthetic_frame, results, tmp_path / "b.png", n_features=3)

    titles = [ax.get_title() for ax in figure.axes]
    expected = list(results.head(3)["feature"])
    for feature, title in zip(expected, titles):
        assert feature in title


def test_the_boxplot_labels_both_classes_on_every_panel(synthetic_frame, tmp_path):
    results = separability_tests(synthetic_frame)

    figure = plot_top_features_boxplots(synthetic_frame, results, tmp_path / "b.png", n_features=2)

    for ax in figure.axes:
        labels = [label.get_text() for label in ax.get_xticklabels()]
        assert len(labels) == 2, "un pannello non distingue le due classi"


def test_asking_for_more_features_than_available_does_not_fail(synthetic_frame, tmp_path):
    results = separability_tests(synthetic_frame)

    figure = plot_top_features_boxplots(synthetic_frame, results, tmp_path / "b.png", n_features=99)

    assert len(figure.axes) == len(results)


def test_the_correlation_heatmap_covers_every_feature(synthetic_frame, tmp_path):
    output = tmp_path / "heatmap.png"

    figure = plot_correlation_heatmap(synthetic_frame, output)

    heatmap_axes = figure.axes[0]
    n_features = len([c for c in synthetic_frame.columns if c not in ("image_name", "category", "target")])
    assert len(heatmap_axes.get_xticks()) == n_features
    assert output.exists()
    assert output.stat().st_size > MIN_PNG_BYTES


def test_the_knn_figure_plots_every_knn_column(synthetic_frame, tmp_path):
    output = tmp_path / "knn.png"

    figure = plot_knn_distributions(synthetic_frame, output)

    assert len(figure.axes) == 4, "attesi i quattro descrittori k-NN"
    assert output.exists()
    assert output.stat().st_size > MIN_PNG_BYTES


def test_the_knn_axes_are_labelled_in_microns(synthetic_frame, tmp_path):
    """Le distanze sono in unita fisiche: l'asse deve dirlo."""
    figure = plot_knn_distributions(synthetic_frame, tmp_path / "knn.png")

    for ax in figure.axes:
        assert "µm" in ax.get_xlabel() or "um" in ax.get_xlabel()


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("n_nuclei", "conteggio"),
        ("nuclear_density_per_1000um2", "nuclei/1000 µm²"),
        ("nuclear_area_fraction", "frazione [0-1]"),
        ("hchannel_mean", "intensità [0-255]"),
        ("area_um2_mean", "µm²"),
        ("knn1_dist_mean_um", "µm"),
        ("circularity_mean", "adimensionale"),
    ],
)
def test_each_feature_is_labelled_with_its_own_unit(feature, expected):
    """Un asse etichettato male e' un errore che in un report passa inosservato."""
    from feature_analysis import _axis_unit

    assert _axis_unit(feature) == expected


def test_the_figures_are_produced_from_the_real_dataset(tmp_path):
    base = Path(__file__).resolve().parent.parent / "data" / "fase3_features"
    if not (base / "features_patches_master.csv").exists():
        pytest.skip("features_patches_master.csv non presente: eseguire la Fase 3.")

    patches = pd.read_csv(base / "features_patches_master.csv")
    results = separability_tests(patches)

    for produce in (
        lambda p: plot_top_features_boxplots(patches, results, p),
        lambda p: plot_correlation_heatmap(patches, p),
        lambda p: plot_knn_distributions(patches, p),
    ):
        output = tmp_path / f"{id(produce)}.png"
        produce(output)
        assert output.stat().st_size > MIN_PNG_BYTES
