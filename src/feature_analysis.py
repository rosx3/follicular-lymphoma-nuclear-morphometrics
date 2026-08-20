"""
===============================================================================
feature_analysis.py — Analisi di Separabilita' Statistica FL vs REACTIVE
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
===============================================================================
Quantifica quali dei 47 biomarcatori estratti dalla Fase 3 distinguono davvero
il linfoma follicolare dal tessuto reattivo, prima e indipendentemente da
qualunque modello di machine learning.

E' il passaggio che giustifica il paradigma white-box: se i biomarcatori fisici
separano le due classi gia' a livello di test statistico univariato, la
successiva classificazione poggia su grandezze interpretabili e non su feature
neurali astratte.

SCELTE METODOLOGICHE
--------------------
1. Test scelto sui dati, non a priori (decisione del piano, Task 6).
   Si verifica la normalita' di ciascun gruppo con Shapiro-Wilk: se entrambi la
   soddisfano si usa il t-test di Welch (che non assume varianze uguali),
   altrimenti Mann-Whitney U. Molte feature morfometriche sono asimmetriche per
   costruzione, quindi imporre il t-test ovunque sarebbe scorretto.

2. Correzione per test multipli (decisione D4).
   Con 47 feature testate a alpha=0.05 ci si attendono ~2 falsi positivi per
   puro caso. Si riporta il p-value grezzo affiancato a quello corretto con
   Benjamini-Hochberg (controllo del False Discovery Rate), e la
   significativita' e' decisa sul p-value corretto.

3. Effect size sempre riportato.
   Con 300 patch per classe anche differenze clinicamente irrilevanti possono
   risultare significative. L'effect size dice quanto e' grande la differenza,
   il p-value solo quanto e' improbabile che sia casuale. Il segno indica la
   direzione: positivo = valore maggiore nel linfoma follicolare.
===============================================================================
"""

from pathlib import Path

import matplotlib

# Backend non interattivo: il modulo produce file, non finestre, e viene
# eseguito anche dalla suite di test dove nessun display e' disponibile.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# Colonne non-feature del CSV della Fase 3. Rispecchia PATCH_METADATA_COLUMNS
# di src/03_feature_extraction.py; un test verifica che le due restino allineate.
METADATA_COLUMNS: tuple[str, ...] = ("image_name", "category", "target")

CATEGORY_FL = "follicular_lymphoma"
CATEGORY_REACTIVE = "reactive_tissue"

# Soglia per il test di normalita' di Shapiro-Wilk.
NORMALITY_ALPHA = 0.05


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Colonne numeriche del DataFrame che rappresentano biomarcatori."""
    return [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]


def _split_by_class(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Valori validi della feature nelle due classi, senza NaN."""
    fl = df.loc[df["category"] == CATEGORY_FL, column].dropna().to_numpy(dtype=float)
    reactive = df.loc[df["category"] == CATEGORY_REACTIVE, column].dropna().to_numpy(dtype=float)
    return fl, reactive


def _is_normal(values: np.ndarray) -> bool:
    """Shapiro-Wilk. Campioni troppo piccoli o costanti: non assumibili normali."""
    if len(values) < 3 or np.ptp(values) == 0.0:
        return False
    return float(stats.shapiro(values).pvalue) > NORMALITY_ALPHA


def _cohens_d(fl: np.ndarray, reactive: np.ndarray) -> float:
    """Differenza fra le medie in unita' di deviazione standard aggregata."""
    n_fl, n_re = len(fl), len(reactive)
    pooled_var = (
        (n_fl - 1) * fl.var(ddof=1) + (n_re - 1) * reactive.var(ddof=1)
    ) / (n_fl + n_re - 2)
    if pooled_var <= 0:
        return 0.0
    return float((fl.mean() - reactive.mean()) / np.sqrt(pooled_var))


def _rank_biserial(u_statistic: float, n_fl: int, n_reactive: int) -> float:
    """
    Correlazione rango-biseriale, effect size non parametrico in [-1, 1].

    Interpretabile come: probabilita' che un valore FL superi un valore
    REACTIVE, riscalata. Positivo = FL tendenzialmente maggiore.
    """
    if n_fl == 0 or n_reactive == 0:
        return 0.0
    return float(2.0 * u_statistic / (n_fl * n_reactive) - 1.0)


def describe_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiche descrittive di ciascuna feature, separate per classe.

    Returns:
        DataFrame indicizzato per feature con media, deviazione standard e
        mediana di entrambe le classi.
    """
    rows = {}
    for column in feature_columns(df):
        fl, reactive = _split_by_class(df, column)
        rows[column] = {
            "mean_fl": fl.mean() if len(fl) else np.nan,
            "std_fl": fl.std(ddof=1) if len(fl) > 1 else np.nan,
            "median_fl": np.median(fl) if len(fl) else np.nan,
            "mean_reactive": reactive.mean() if len(reactive) else np.nan,
            "std_reactive": reactive.std(ddof=1) if len(reactive) > 1 else np.nan,
            "median_reactive": np.median(reactive) if len(reactive) else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def separability_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Testa la separabilita' FL vs REACTIVE di ogni feature.

    Args:
        df: CSV per patch della Fase 3, con la colonna `category`.
        alpha: soglia di significativita', applicata al p-value corretto FDR.

    Returns:
        DataFrame con una riga per feature, ordinato per evidenza decrescente:
        feature, n per classe, medie, test usato, statistica, p_raw, p_fdr,
        effect_size (segno positivo = maggiore in FL), effect_size_type,
        significant.
    """
    records = []

    for column in feature_columns(df):
        fl, reactive = _split_by_class(df, column)

        # Una feature costante o priva di dati non e' separabile: dichiararlo
        # esplicitamente evita che scipy restituisca NaN o sollevi.
        combined = np.concatenate([fl, reactive]) if len(fl) and len(reactive) else np.array([])
        if len(fl) < 3 or len(reactive) < 3 or (len(combined) and np.ptp(combined) == 0.0):
            records.append({
                "feature": column,
                "n_fl": len(fl),
                "n_reactive": len(reactive),
                "mean_fl": fl.mean() if len(fl) else np.nan,
                "mean_reactive": reactive.mean() if len(reactive) else np.nan,
                "test": "non_applicabile",
                "statistic": np.nan,
                "p_raw": 1.0,
                "effect_size": 0.0,
                "effect_size_type": "nessuno",
            })
            continue

        if _is_normal(fl) and _is_normal(reactive):
            # Welch: non assume varianze uguali fra i due gruppi.
            result = stats.ttest_ind(fl, reactive, equal_var=False)
            test_name, effect, effect_type = "welch_t", _cohens_d(fl, reactive), "cohens_d"
        else:
            result = stats.mannwhitneyu(fl, reactive, alternative="two-sided")
            test_name = "mann_whitney_u"
            effect = _rank_biserial(float(result.statistic), len(fl), len(reactive))
            effect_type = "rank_biserial"

        records.append({
            "feature": column,
            "n_fl": len(fl),
            "n_reactive": len(reactive),
            "mean_fl": float(fl.mean()),
            "mean_reactive": float(reactive.mean()),
            "test": test_name,
            "statistic": float(result.statistic),
            "p_raw": float(result.pvalue),
            "effect_size": effect,
            "effect_size_type": effect_type,
        })

    results = pd.DataFrame.from_records(records)
    if results.empty:
        return results

    # Benjamini-Hochberg sul FDR (decisione D4). scipy>=1.11 lo fornisce
    # nativamente: nessuna dipendenza aggiuntiva da statsmodels.
    results["p_fdr"] = stats.false_discovery_control(results["p_raw"].to_numpy(), method="bh")
    results["significant"] = results["p_fdr"] < alpha

    results = results[[
        "feature", "n_fl", "n_reactive", "mean_fl", "mean_reactive",
        "test", "statistic", "p_raw", "p_fdr", "effect_size",
        "effect_size_type", "significant",
    ]]

    # Ordinamento per evidenza: prima il p-value corretto, poi l'effect size.
    return (
        results.sort_values(
            ["p_fdr", "effect_size"],
            key=lambda s: s.abs() if s.name == "effect_size" else s,
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Figure per la tesi
# ---------------------------------------------------------------------------
FIGURE_DPI = 300

# Palette: rosso per il patologico, blu per il reattivo, coerente con le
# anteprime della Fase 2.
CLASS_COLORS = {CATEGORY_FL: "#c0392b", CATEGORY_REACTIVE: "#2471a3"}
CLASS_LABELS = {CATEGORY_FL: "FL", CATEGORY_REACTIVE: "REACTIVE"}

# Unita' di misura per etichettare gli assi, in ordine di specificita': la
# prima regola che corrisponde vince, quindi i casi particolari precedono i
# suffissi generici (_um2 va valutato prima di _um, che ne e' un prefisso).
_UNIT_RULES: tuple[tuple[str, str], ...] = (
    ("_per_1000um2", "nuclei/1000 µm²"),
    ("_fraction", "frazione [0-1]"),
    ("_um2", "µm²"),
    ("_um", "µm"),
)


def _axis_unit(feature: str) -> str:
    """
    Unita' di misura di una feature, dedotta dal nome.

    Un asse etichettato male e' un errore che in un report passa inosservato:
    ogni famiglia di feature ha la sua unita' e le adimensionali vanno
    dichiarate tali, non lasciate senza etichetta.
    """
    if feature == "n_nuclei":
        return "conteggio"
    if feature.startswith("hchannel_"):
        return "intensità [0-255]"
    for suffix, unit in _UNIT_RULES:
        if suffix in feature:
            return unit
    return "adimensionale"


def _save(figure: plt.Figure, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    return output_path


def plot_top_features_boxplots(
    df: pd.DataFrame, results: pd.DataFrame, output_path: Path, n_features: int = 6
) -> plt.Figure:
    """
    Boxplot affiancati FL vs REACTIVE per le feature con l'evidenza piu forte.

    Args:
        df: matrice per patch della Fase 3.
        results: output di separability_tests(), gia' ordinato per evidenza.
        n_features: quanti pannelli produrre (limitato alle feature disponibili).
    """
    selected = list(results.head(n_features)["feature"])
    n_cols = min(3, len(selected))
    n_rows = int(np.ceil(len(selected) / n_cols))

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, selected):
        groups = [
            df.loc[df["category"] == category, feature].dropna()
            for category in (CATEGORY_FL, CATEGORY_REACTIVE)
        ]
        boxes = ax.boxplot(groups, patch_artist=True, widths=0.55, showfliers=False)
        for patch, category in zip(boxes["boxes"], (CATEGORY_FL, CATEGORY_REACTIVE)):
            patch.set_facecolor(CLASS_COLORS[category])
            patch.set_alpha(0.65)
        for median in boxes["medians"]:
            median.set_color("black")

        row = results.set_index("feature").loc[feature]
        ax.set_xticks([1, 2])
        ax.set_xticklabels([CLASS_LABELS[CATEGORY_FL], CLASS_LABELS[CATEGORY_REACTIVE]])
        ax.set_title(f"{feature}\nFDR p = {row['p_fdr']:.2e}", fontsize=10, fontweight="bold")
        ax.set_ylabel(_axis_unit(feature), fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Gli assi in eccesso della griglia vanno rimossi, non solo nascosti: un
    # asse invisibile resta in figure.axes e la figura non corrisponderebbe piu'
    # a cio' che mostra.
    for ax in axes[len(selected):]:
        figure.delaxes(ax)

    figure.suptitle(
        "Biomarcatori piu discriminanti — Linfoma Follicolare vs Tessuto Reattivo",
        fontsize=13, fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    _save(figure, output_path)
    return figure


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> plt.Figure:
    """
    Mappa di correlazione di Spearman fra tutte le feature.

    Serve ad anticipare la multicollinearita' che distorce l'interpretazione
    SHAP in Fase 4: fra due feature quasi identiche l'importanza viene divisa
    arbitrariamente fra le due.
    """
    columns = feature_columns(df)
    correlation = df[columns].corr(method="spearman")

    figure, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=6)
    ax.set_yticklabels(columns, fontsize=6)
    ax.set_title(
        "Correlazione di Spearman fra i 47 biomarcatori\n"
        "(rosso = correlazione positiva, blu = negativa)",
        fontsize=12, fontweight="bold",
    )

    colorbar = figure.colorbar(image, ax=ax, shrink=0.75)
    colorbar.set_label("rho di Spearman", fontsize=9)

    figure.tight_layout()
    _save(figure, output_path)
    return figure


def plot_knn_distributions(df: pd.DataFrame, output_path: Path) -> plt.Figure:
    """
    Distribuzioni dei quattro descrittori k-NN per classe.

    E' il risultato micro-spaziale caratteristico di questo lavoro: sostituisce
    i grafi di Delaunay/MST, esclusi per la scala della patch (report §1.1).
    """
    knn_columns = [c for c in feature_columns(df) if c.startswith("knn")]

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for ax, feature in zip(axes, knn_columns):
        for category in (CATEGORY_FL, CATEGORY_REACTIVE):
            values = df.loc[df["category"] == category, feature].dropna()
            ax.hist(
                values, bins=30, alpha=0.55, label=CLASS_LABELS[category],
                color=CLASS_COLORS[category], edgecolor="none",
            )
            ax.axvline(values.mean(), color=CLASS_COLORS[category], linestyle="--", linewidth=1.4)

        ax.set_title(feature, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"distanza [{_axis_unit(feature)}]", fontsize=9)
        ax.set_ylabel("numero di patch", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes[len(knn_columns):]:
        figure.delaxes(ax)

    figure.suptitle(
        "Distribuzione delle distanze inter-nucleari (k-NN)\n"
        "linea tratteggiata = media di classe",
        fontsize=13, fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    _save(figure, output_path)
    return figure


# ---------------------------------------------------------------------------
# Esecuzione: genera separability_tests.csv dalla matrice della Fase 3
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    fase3_dir = base_dir / "data" / "fase3_features"

    patches = pd.read_csv(fase3_dir / "features_patches_master.csv")
    print(f"[Analisi] Matrice caricata: {patches.shape[0]} patch x {patches.shape[1]} colonne")

    results = separability_tests(patches)
    output_path = fase3_dir / "separability_tests.csv"
    results.to_csv(output_path, index=False)

    n_significant = int(results["significant"].sum())
    print(f"[Analisi] Feature significative (FDR < 0.05): {n_significant} / {len(results)}")
    print(f"[Analisi] Test di separabilita' -> {output_path}")

    img_dir = base_dir / "img" / "fase3"
    for produce, name in (
        (lambda p: plot_top_features_boxplots(patches, results, p), "boxplot_top_features.png"),
        (lambda p: plot_correlation_heatmap(patches, p), "correlation_heatmap.png"),
        (lambda p: plot_knn_distributions(patches, p), "knn_distribution.png"),
    ):
        path = img_dir / name
        produce(path)
        plt.close("all")
        print(f"[Analisi] Figura -> {path} ({path.stat().st_size // 1024} KB)")
