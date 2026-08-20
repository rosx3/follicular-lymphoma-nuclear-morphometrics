"""
===============================================================================
Modulo 04: Classificazione Machine Learning Tabulare & XAI
Tesi: Classificazione Linfoma Follicolare vs Tessuto Reattivo
===============================================================================
Spec: docs/superpowers/specs/2026-08-20-fase4-classificazione-design.md

Classifica le 600 patch a partire dai biomarcatori della Fase 3 e spiega quali
guidano la decisione. La spiegabilita' non e' un complemento: e' la ragione per
cui la tesi adotta un approccio white-box invece di una CNN end-to-end.

IL PROBLEMA CENTRALE — perche' ci sono due validazioni e non una.

  Le 600 patch non vengono da 600 pazienti. La serie di origine (Carreras et al.
  2025) e' di 221 casi, da cui sono state estratte ~1,5 milioni di patch: piu'
  patch per caso, quindi. Il dataset pubblicato su Zenodo e' pero' piatto e non
  contiene identificativi di caso: l'informazione non e' recuperabile.

  Con uno split casuale, patch dello stesso vetrino finiscono sia in
  addestramento sia in test, e il modello puo' imparare la firma del vetrino —
  intensita' della colorazione, resa dello scanner — invece della biologia. Le
  feature piu' esposte sono proprio quelle di tessitura e intensita'
  (hchannel_mean, glcm_*, lbp_entropy). Il punteggio risponderebbe allora a una
  domanda piu' facile e clinicamente inutile: "riconosco altre patch di pazienti
  che ho gia' visto".

  Che il rischio sia reale su questi dati e' stato verificato: patch di indice
  adiacente distano nello spazio dei biomarcatori 0.615x (FL) e 0.691x
  (REACTIVE) rispetto a coppie qualsiasi. L'ordine di numerazione conserva una
  struttura a blocchi compatibile con l'esportazione caso per caso.

  Ogni modello viene percio' valutato DUE volte, stessi dati e stesso codice,
  cambiando solo lo splitter: casuale (ottimistico) e a blocchi contigui
  (conservativo). Cio' che si pubblica e' la forbice fra i due.
===============================================================================
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_INDEX_PATTERN = re.compile(r"^(?P<prefix>.*?)\s*\((?P<index>\d+)\)\s*$")

# Colonne di metadato del CSV della Fase 3: identificano la patch ma non sono
# biomarcatori e non devono mai entrare nella matrice di addestramento.
METADATA_COLUMNS = ("image_name", "category", "target")


# ---------------------------------------------------------------------------
# D3 — classifica di leggibilita' clinica
#
# Fra biomarcatori quasi identici (|rho| oltre soglia) se ne tiene uno solo, e a
# decidere e' questa classifica: dalla grandezza che un patologo nomina e misura
# direttamente, giu' fino ai momenti statistici di ordine superiore.
#
# Il criterio alternativo — tenere la variabile con l'effect size maggiore —
# sarebbe piu' oggettivo ma su questi dati sceglie sistematicamente la variabile
# meno comprensibile: terrebbe knn3_dist_mean_um scartando n_nuclei, e
# circularity_skew scartando circularity_mean. Poiche' l'output della Fase 4 e'
# una spiegazione clinica, qui prevale la leggibilita'. La scelta e' registrata
# gruppo per gruppo in feature_reduction.csv, cosi' il lettore puo' dissentire
# con cognizione.
# ---------------------------------------------------------------------------
_READABILITY_TIERS: tuple[tuple[str, ...], ...] = (
    # 1 — Grandezze misurabili e nominabili direttamente sul vetrino
    (
        "n_nuclei", "nuclear_area_fraction", "area_um2_mean", "area_top10_mean_um2",
        "area_top10_short_axis_um", "perimeter_um_mean", "major_axis_um_mean",
        "minor_axis_um_mean", "circularity_mean", "eccentricity_mean",
        "solidity_mean", "aspect_ratio_mean", "knn1_dist_mean_um",
        "knn3_dist_mean_um", "hchannel_mean",
    ),
    # 2 — Dispersione della stessa grandezza: "quanto variano fra loro i nuclei"
    (
        "area_um2_std", "perimeter_um_std", "major_axis_um_std", "minor_axis_um_std",
        "circularity_std", "eccentricity_std", "solidity_std", "aspect_ratio_std",
        "knn1_dist_std_um", "knn3_dist_std_um", "hchannel_std",
    ),
    # 3 — Grandezze derivate o di tessitura: interpretabili, ma non misurabili
    #     a occhio su un vetrino
    (
        "nuclear_density_per_1000um2", "glcm_contrast", "glcm_homogeneity",
        "glcm_energy", "lbp_entropy",
    ),
    # 4 — Dispersione normalizzata (coefficiente di variazione)
    (
        "area_um2_cv", "perimeter_um_cv", "circularity_cv", "eccentricity_cv",
        "solidity_cv", "major_axis_um_cv", "minor_axis_um_cv", "aspect_ratio_cv",
    ),
    # 5 — Momenti di ordine superiore: nessun referente clinico immediato
    (
        "area_um2_skew", "perimeter_um_skew", "circularity_skew", "eccentricity_skew",
        "solidity_skew", "major_axis_um_skew", "minor_axis_um_skew", "aspect_ratio_skew",
    ),
)

READABILITY_ORDER: tuple[str, ...] = tuple(f for tier in _READABILITY_TIERS for f in tier)


def _readability_rank(feature: str) -> int:
    """Posizione in classifica; le feature non censite finiscono in fondo."""
    try:
        return READABILITY_ORDER.index(feature)
    except ValueError:
        return len(READABILITY_ORDER)


@dataclass(frozen=True)
class FeatureReduction:
    """Esito della riduzione: cosa resta, e chi rappresenta cosa."""

    kept_features: list[str]
    assignments: list[dict]  # feature, group, kept, representative


def reduce_redundant_features(
    X: np.ndarray, feature_names: list[str], threshold: float = 0.90
) -> FeatureReduction:
    """
    Raggruppa i biomarcatori quasi identici e ne tiene uno per gruppo.

    Gli alberi tollerano la collinearita', ma SHAP no: fra due variabili quasi
    identiche il merito viene diviso arbitrariamente ed entrambe appaiono meno
    importanti di quanto sono. Poiche' la spiegazione e' l'obiettivo della fase,
    le ridondanze si riducono prima di addestrare.

    Args:
        X: matrice (n_patch, n_feature).
        feature_names: nomi delle colonne di X.
        threshold: due feature stanno nello stesso gruppo se |rho| lo supera.

    Returns:
        FeatureReduction con le feature tenute (nell'ordine originale) e, per
        ognuna delle 47, il gruppo di appartenenza e chi la rappresenta.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    if not 0 < threshold <= 1:
        raise ValueError(f"threshold deve stare in (0, 1], ricevuto {threshold}")

    rho = np.abs(spearmanr(X).statistic)
    rho = np.atleast_2d(rho)
    distance = 1.0 - rho
    np.fill_diagonal(distance, 0.0)
    # La matrice di Spearman puo' non essere perfettamente simmetrica per
    # arrotondamento: squareform lo esige.
    distance = (distance + distance.T) / 2.0

    labels = fcluster(
        linkage(squareform(distance, checks=False), method="average"),
        1.0 - threshold,
        criterion="distance",
    )

    representative_of = {}
    for group_id in np.unique(labels):
        members = [feature_names[i] for i in np.where(labels == group_id)[0]]
        representative_of[group_id] = min(members, key=_readability_rank)

    assignments = [
        {
            "feature": name,
            "group": int(group_id),
            "kept": representative_of[group_id] == name,
            "representative": representative_of[group_id],
        }
        for name, group_id in zip(feature_names, labels)
    ]

    return FeatureReduction(
        kept_features=[row["feature"] for row in assignments if row["kept"]],
        assignments=assignments,
    )


@dataclass(frozen=True)
class FeatureMatrix:
    """Matrice dei biomarcatori pronta per la validazione."""

    X: np.ndarray            # (n_patch, n_feature)
    y: np.ndarray            # (n_patch,) 1 = linfoma follicolare, 0 = reattivo
    feature_names: list[str]
    image_names: list[str]   # serve a ricostruire i blocchi contigui


def load_feature_matrix(csv_path) -> FeatureMatrix:
    """
    Legge la matrice della Fase 3 e la prepara per la Fase 4.

    Raises:
        FileNotFoundError: se il CSV non esiste.
        ValueError: se mancano colonne attese o se compaiono valori mancanti.
            La Fase 3 garantisce zero NaN: se ne comparissero, qualcosa a monte
            e' cambiato e va indagato, non imputato in silenzio.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} non trovato: eseguire prima la Fase 3 con "
            "`python src/run_pipeline.py --fase 3`."
        )

    table = pd.read_csv(csv_path)

    missing = [column for column in METADATA_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"colonne di metadato mancanti in {csv_path.name}: {missing}")

    feature_names = [c for c in table.columns if c not in METADATA_COLUMNS]
    if not feature_names:
        raise ValueError(f"{csv_path.name} non contiene alcun biomarcatore")

    X = table[feature_names].to_numpy(dtype=float)
    if np.isnan(X).any():
        incomplete = [f for f in feature_names if table[f].isna().any()]
        raise ValueError(
            f"valori mancanti in {csv_path.name}, colonne: {incomplete}. La Fase 3 "
            "non dovrebbe produrne: indagare a monte invece di imputare."
        )

    return FeatureMatrix(
        X=X,
        y=table["target"].to_numpy(dtype=int),
        feature_names=feature_names,
        image_names=table["image_name"].astype(str).tolist(),
    )


def contiguous_blocks(image_names, block_size: int) -> list[str]:
    """
    Assegna a ogni patch un identificativo di blocco, sostituto delle etichette
    di caso che il dataset non contiene.

    I blocchi raggruppano patch con indici vicini all'interno della stessa
    categoria: e' l'approssimazione piu' prudente disponibile del "stesso caso".
    Non pretendono di identificare i pazienti — servono come sonda per misurare
    quanto il punteggio dipenda dal vicinato.

    L'appartenenza si basa sulla POSIZIONE nell'ordine numerico, non sul valore
    dell'indice: cosi' eventuali buchi nella numerazione non producono blocchi di
    dimensione irregolare.

    Args:
        image_names: nomi delle patch, nella forma "<prefisso> (<indice>)".
            Il prefisso identifica la categoria e non viene mai attraversato da
            un blocco.
        block_size: quante patch consecutive raggruppare.

    Returns:
        Lista di identificativi di blocco, nello stesso ordine di `image_names`.

    Raises:
        ValueError: se un nome non contiene un indice numerico. La validazione a
            blocchi dipende da quell'ordine: un nome non conforme e' un errore
            visibile, non una patch da assegnare a caso.
    """
    if block_size < 1:
        raise ValueError(f"block_size deve essere >= 1, ricevuto {block_size}")

    parsed = []
    for position, name in enumerate(image_names):
        match = _INDEX_PATTERN.match(str(name))
        if match is None:
            raise ValueError(
                f"impossibile ricavare l'indice numerico da '{name}': la validazione "
                "a blocchi richiede nomi nella forma '<prefisso> (<numero>)'."
            )
        parsed.append((match["prefix"], int(match["index"]), position))

    blocks: list[str | None] = [None] * len(parsed)
    for prefix in {prefix for prefix, _, _ in parsed}:
        ordered = sorted((idx, pos) for pfx, idx, pos in parsed if pfx == prefix)
        for rank, (_, position) in enumerate(ordered):
            blocks[position] = f"{prefix}#{rank // block_size}"

    return blocks


# ---------------------------------------------------------------------------
# D2 — i tre modelli a confronto
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Uno stimatore con la griglia su cui tararlo."""

    estimator: object       # sklearn Pipeline
    param_grid: dict


def build_models(seed: int = 42) -> dict[str, ModelSpec]:
    """
    Regressione logistica, Random Forest e XGBoost, ciascuno in una Pipeline.

    La logistica e' il riferimento minimo: se regge il confronto con gli alberi,
    la complessita' non serve su questi dati — ed e' un risultato da riportare,
    non una sconfitta.

    La standardizzazione vive DENTRO la Pipeline, cosi' viene adattata solo sui
    dati di addestramento di ogni piega. Adattarla prima dello split sarebbe
    leakage, della stessa famiglia di D1. Gli alberi non la ricevono: sono
    invarianti a riscalature (Fase 3, §5) e scalarli aggiungerebbe solo rumore.

    Le griglie sono deliberatamente piccole: con 600 righe una ricerca ampia
    sovradatta la ricerca stessa, e il costo si moltiplica per ogni validazione.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    return {
        "logistic_regression": ModelSpec(
            estimator=Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=seed)),
            ]),
            param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": ModelSpec(
            estimator=Pipeline([
                ("model", RandomForestClassifier(random_state=seed, n_jobs=-1)),
            ]),
            param_grid={
                "model__n_estimators": [300],
                "model__max_depth": [None, 6],
                "model__min_samples_leaf": [1, 5],
            },
        ),
        "xgboost": ModelSpec(
            estimator=Pipeline([
                ("model", XGBClassifier(
                    random_state=seed, n_jobs=-1, eval_metric="logloss",
                    tree_method="hist",
                )),
            ]),
            param_grid={
                "model__n_estimators": [300],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.05, 0.2],
            },
        ),
    }


# ---------------------------------------------------------------------------
# D4/D5 — valutazione con taratura annidata
# ---------------------------------------------------------------------------
def _fold_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    """AUC-ROC, accuratezza bilanciata, sensibilita' e specificita'."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # Sensibilita': quanti linfomi vengono riconosciuti. Specificita': quanti
        # tessuti reattivi non vengono scambiati per linfoma.
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
    }


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    models: dict[str, ModelSpec],
    splitter,
    validation: str,
    seed: int = 42,
    groups=None,
    inner_splits: int = 3,
    return_predictions: bool = False,
) -> pd.DataFrame:
    """
    Valuta ogni modello con cross-validation annidata.

    Gli iperparametri si scelgono dentro ogni piega di addestramento, mai
    sull'intero dataset: sceglierli guardando anche i dati di test gonfierebbe
    le metriche esattamente come il leakage di D1, applicato ai parametri invece
    che alle righe.

    Quando `groups` e' fornito, anche la ricerca interna e' raggruppata: se lo
    split esterno separa i blocchi ma quello interno li mescola, la selezione
    degli iperparametri torna a poggiare su dati contaminati.

    Args:
        splitter: lo splitter esterno (StratifiedKFold o GroupKFold).
        validation: etichetta della validazione, finisce nella colonna omonima.
        groups: identificativi di blocco, necessari per GroupKFold.

    Returns:
        DataFrame con una riga per modello e piega. Con `return_predictions`,
        anche le probabilita' fuori-piega: servono alle curve ROC e all'analisi
        di dove il modello sbaglia, senza dover rieseguire l'intera valutazione.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold

    groups = None if groups is None else np.asarray(groups)
    split_args = (X, y) if groups is None else (X, y, groups)

    records, predictions = [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # I blocchi non mescolano mai le categorie, quindi uno splitter non
        # stratificato puo' produrre una piega monoclasse. L'AUC sarebbe
        # indefinita e, mediata in silenzio, darebbe un numero falso.
        for subset_name, subset in (("test", y_test), ("addestramento", y_train)):
            if len(np.unique(subset)) < 2:
                raise ValueError(
                    f"piega {fold}: il sottoinsieme di {subset_name} contiene una "
                    "sola classe, l'AUC non e' definita. Ridurre la dimensione dei "
                    "blocchi o il numero di pieghe."
                )

        if groups is None:
            inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
            inner_fit_params = {}
        else:
            inner = GroupKFold(n_splits=inner_splits)
            inner_fit_params = {"groups": groups[train_idx]}

        for name, spec in models.items():
            search = GridSearchCV(
                clone(spec.estimator),
                spec.param_grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train, **inner_fit_params)

            # Stessa insidia un livello piu' sotto: se una piega INTERNA e'
            # monoclasse, GridSearchCV assegna nan e sceglie gli iperparametri
            # praticamente a caso, senza che nulla lo segnali.
            if not np.isfinite(search.best_score_):
                raise ValueError(
                    f"piega {fold}, modello {name}: la ricerca interna non ha "
                    "prodotto punteggi finiti — probabile piega interna con una "
                    "sola classe. Ridurre inner_splits o la dimensione dei blocchi."
                )

            y_prob = search.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            records.append({
                "validation": validation,
                "model": name,
                "fold": fold,
                **_fold_metrics(y_test, y_prob, y_pred),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "best_params": str(search.best_params_),
            })
            predictions.append(pd.DataFrame({
                "validation": validation, "model": name, "fold": fold,
                "row": test_idx, "y_true": y_test, "y_prob": y_prob,
            }))

    metrics = pd.DataFrame(records)
    return (metrics, pd.concat(predictions, ignore_index=True)) if return_predictions else metrics


def block_size_sensitivity(
    X: np.ndarray,
    y: np.ndarray,
    image_names: list[str],
    models: dict[str, ModelSpec],
    block_sizes=(5, 10, 20, 30),
    seed: int = 42,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Ripete la validazione a blocchi al variare della dimensione del blocco.

    Serve a non far dipendere la conclusione da un parametro arbitrario. Se la
    metrica degrada al crescere del blocco, la dipendenza dal vicinato e' reale;
    se resta piatta, i blocchi non catturavano struttura e le due validazioni
    vanno considerate equivalenti — conclusione altrettanto valida, purche'
    dichiarata.
    """
    from sklearn.model_selection import GroupKFold

    tables = []
    for block_size in block_sizes:
        groups = np.array(contiguous_blocks(image_names, block_size=block_size))
        table = evaluate(
            X, y, models,
            GroupKFold(n_splits=n_splits),
            validation=f"B_blocchi_{block_size}",
            seed=seed,
            groups=groups,
        )
        table["block_size"] = block_size
        table["n_blocks"] = len(set(groups))
        tables.append(table)

    return pd.concat(tables, ignore_index=True)


# ---------------------------------------------------------------------------
# D6 — spiegazione con SHAP
# ---------------------------------------------------------------------------
def explain_with_shap(model, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Importanza globale dei biomarcatori secondo SHAP, con la direzione dell'effetto.

    Args:
        model: Pipeline gia' addestrata, il cui ultimo passo e' un modello ad albero.
        X: matrice su cui calcolare le spiegazioni.
        feature_names: nomi delle colonne di X.

    Returns:
        DataFrame ordinato per importanza decrescente, con:
          - `importance`: media dei valori assoluti SHAP;
          - `direction`: correlazione fra valore della feature e suo contributo.
            Positiva = valori alti spingono verso il linfoma follicolare.
    """
    import shap

    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    transformed = model[:-1].transform(X) if hasattr(model, "steps") and len(model.steps) > 1 else X

    values = shap.TreeExplainer(estimator).shap_values(transformed)
    values = np.asarray(values)
    # Alcune versioni restituiscono (n, k, 2) per la classificazione binaria:
    # si tiene la classe positiva.
    if values.ndim == 3:
        values = values[:, :, 1]

    directions, rhos, profiles = [], [], []
    for column in range(values.shape[1]):
        feature_column = np.asarray(transformed)[:, column]
        direction, rho, profile = _effect_direction(feature_column, values[:, column])
        directions.append(direction)
        rhos.append(rho)
        profiles.append(profile)

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(values).mean(axis=0),
            "direction": directions,
            "direction_rho": rhos,
            "quintile_profile": profiles,
        })
        .sort_values("importance", ascending=False, ignore_index=True)
    )


# Un contributo puo' variare col valore senza farlo in modo monotono. Sotto
# questa frazione dell'escursione totale, una variazione fra quintili consecutivi
# e' considerata rumore e non rompe la monotonia.
_MONOTONICITY_TOLERANCE = 0.10


def _effect_direction(feature_values: np.ndarray, shap_values: np.ndarray) -> tuple[str, float, str]:
    """
    Direzione dell'effetto, dichiarata solo quando ha senso dichiararla.

    Una correlazione lineare fra valore e contributo descrive bene un effetto
    monotono e mente su un effetto a U. Su questi dati il caso esiste:
    `solidity_mean` ha medie di classe quasi identiche (Mann-Whitney p = 0.106)
    ma dispersioni molto diverse (Levene p = 3.0e-06), e valori estremi in
    ENTRAMBE le direzioni spingono verso il linfoma. Attribuirle una direzione
    metterebbe un'affermazione falsa nella tesi.

    Si guarda percio' il profilo del contributo medio lungo i quintili del
    valore: se sale (o scende) senza inversioni degne di nota la direzione e'
    dichiarabile, altrimenti si dichiara che non lo e'.

    Returns:
        (direzione, rho di Spearman, profilo per quintile leggibile).
        La direzione e' "FL", "REACTIVE" oppure "non monotona".
    """
    from scipy.stats import spearmanr

    if np.std(feature_values) == 0 or np.std(shap_values) == 0:
        return "non monotona", 0.0, ""

    rho = float(spearmanr(feature_values, shap_values).statistic)

    quintiles = pd.qcut(feature_values, 5, labels=False, duplicates="drop")
    means = np.array([shap_values[quintiles == q].mean() for q in range(int(quintiles.max()) + 1)])
    profile = " ".join(f"{value:+.2f}" for value in means)

    span = float(means.max() - means.min())
    if span == 0:
        return "non monotona", rho, profile

    steps = np.diff(means)
    meaningful = steps[np.abs(steps) > _MONOTONICITY_TOLERANCE * span]
    if meaningful.size == 0 or not (np.all(meaningful > 0) or np.all(meaningful < 0)):
        return "non monotona", rho, profile

    # Contributo positivo = spinta verso la classe 1, cioe' il linfoma follicolare.
    return ("FL" if meaningful[0] > 0 else "REACTIVE"), rho, profile


# ---------------------------------------------------------------------------
# Orchestrazione
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_CSV = BASE_DIR / "data" / "fase3_features" / "features_patches_master.csv"
SEPARABILITY_CSV = BASE_DIR / "data" / "fase3_features" / "separability_tests.csv"
OUTPUT_DIR = BASE_DIR / "data" / "fase4_classification"
IMG_DIR = BASE_DIR / "img" / "fase4"

SEED = 42
REDUNDANCY_THRESHOLD = 0.90
MAIN_BLOCK_SIZE = 10          # scelto prima di vedere le metriche (spec, D1)
BLOCK_SIZES = (5, 10, 20, 30)
N_SPLITS = 5


def _plot_roc(predictions: pd.DataFrame, path: Path) -> None:
    """Curve ROC fuori-piega, un pannello per validazione."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    validations = list(dict.fromkeys(predictions["validation"]))
    fig, axes = plt.subplots(1, len(validations), figsize=(11, 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, validation in zip(axes, validations):
        subset = predictions[predictions.validation == validation]
        for model in dict.fromkeys(subset["model"]):
            rows = subset[subset.model == model]
            fpr, tpr, _ = roc_curve(rows["y_true"], rows["y_prob"])
            auc = roc_auc_score(rows["y_true"], rows["y_prob"])
            ax.plot(fpr, tpr, label=f"{model} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(validation)
        ax.set_xlabel("1 - specificita")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("sensibilita")
    fig.suptitle("Curve ROC fuori-piega — validazione casuale contro validazione a blocchi")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_validation_gap(metrics: pd.DataFrame, path: Path) -> None:
    """La forbice: quanto del punteggio spariva passando ai blocchi."""
    import matplotlib.pyplot as plt

    summary = metrics.groupby(["model", "validation"])["auc_roc"].agg(["mean", "std"])
    models = list(dict.fromkeys(metrics["model"]))
    validations = list(dict.fromkeys(metrics["validation"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    positions = np.arange(len(models))
    for offset, validation in zip((-width / 2, width / 2), validations):
        means = [summary.loc[(m, validation), "mean"] for m in models]
        errors = [summary.loc[(m, validation), "std"] for m in models]
        ax.bar(positions + offset, means, width, yerr=errors, capsize=4, label=validation)

    for position, model in zip(positions, models):
        gap = (summary.loc[(model, validations[0]), "mean"]
               - summary.loc[(model, validations[1]), "mean"])
        ax.text(position, 1.005, f"forbice {gap:+.3f}", ha="center", fontsize=9)

    ax.set_xticks(positions, models, rotation=10)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Quanto del punteggio era leakage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_shap_summary(model, X: np.ndarray, feature_names: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt
    import shap

    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    values = np.asarray(shap.TreeExplainer(estimator).shap_values(X))
    if values.ndim == 3:
        values = values[:, :, 1]

    plt.figure(figsize=(9, 8))
    shap.summary_plot(values, X, feature_names=feature_names, show=False, max_display=20)
    plt.title("Biomarcatori che decidono la classificazione (SHAP)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_shap_vs_univariate(importance: pd.DataFrame, path: Path) -> None:
    """
    Confronto fra la gerarchia SHAP e gli effect size univariati della Fase 3.

    Una divergenza non e' di per se' un errore — una variabile debole da sola puo'
    contare in combinazione — ma va vista, non subita.
    """
    import matplotlib.pyplot as plt

    if not SEPARABILITY_CSV.exists():
        return
    separability = pd.read_csv(SEPARABILITY_CSV).set_index("feature")
    merged = importance.copy()
    merged["effect_size"] = merged["feature"].map(separability["effect_size"].abs())
    merged = merged.dropna(subset=["effect_size"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["effect_size"], merged["importance"], s=30)
    for _, row in merged.head(8).iterrows():
        ax.annotate(row["feature"], (row["effect_size"], row["importance"]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("|effect size| univariato (Fase 3)")
    ax.set_ylabel("importanza SHAP (multivariata)")
    ax.set_title("La gerarchia SHAP concorda con i test univariati?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Esegue l'intera Fase 4 e scrive gli artefatti dichiarati nella spec."""
    import json
    import platform

    import joblib
    import matplotlib
    import sklearn
    import xgboost
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    matplotlib.use("Agg")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("[Fase 4] Caricamento della matrice dei biomarcatori...")
    data = load_feature_matrix(MASTER_CSV)

    print(f"[Fase 4] Riduzione delle ridondanze (|rho| > {REDUNDANCY_THRESHOLD})...")
    reduction = reduce_redundant_features(data.X, data.feature_names, REDUNDANCY_THRESHOLD)
    pd.DataFrame(reduction.assignments).to_csv(OUTPUT_DIR / "feature_reduction.csv", index=False)
    columns = [data.feature_names.index(f) for f in reduction.kept_features]
    X = data.X[:, columns]
    print(f"          {len(data.feature_names)} -> {len(reduction.kept_features)} biomarcatori")

    models = build_models(seed=SEED)
    groups = np.array(contiguous_blocks(data.image_names, block_size=MAIN_BLOCK_SIZE))

    print("[Fase 4] Validazione A — split casuale (stima ottimistica)...")
    metrics_a, predictions_a = evaluate(
        X, data.y, models,
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
        validation="A_casuale", seed=SEED, return_predictions=True,
    )
    print(f"[Fase 4] Validazione B — blocchi da {MAIN_BLOCK_SIZE} (stima conservativa)...")
    metrics_b, predictions_b = evaluate(
        X, data.y, models, GroupKFold(n_splits=N_SPLITS),
        validation="B_blocchi", seed=SEED, groups=groups, return_predictions=True,
    )
    metrics = pd.concat([metrics_a, metrics_b], ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "metrics_by_model.csv", index=False)

    predictions = pd.concat([predictions_a, predictions_b], ignore_index=True)
    predictions["image_name"] = predictions["row"].map(dict(enumerate(data.image_names)))
    predictions["block"] = predictions["row"].map(dict(enumerate(groups)))
    predictions.to_csv(OUTPUT_DIR / "out_of_fold_predictions.csv", index=False)

    print(f"[Fase 4] Sensibilita' alla dimensione del blocco {BLOCK_SIZES}...")
    sensitivity = block_size_sensitivity(
        X, data.y, data.image_names, models,
        block_sizes=BLOCK_SIZES, seed=SEED, n_splits=N_SPLITS,
    )
    sensitivity.to_csv(OUTPUT_DIR / "block_size_sensitivity.csv", index=False)

    # Il modello migliore si sceglie sulla validazione CONSERVATIVA: sceglierlo
    # su quella ottimistica premierebbe chi sfrutta meglio il leakage.
    ranking = metrics_b.groupby("model")["auc_roc"].mean().sort_values(ascending=False)
    best_name = str(ranking.index[0])
    print(f"[Fase 4] Modello migliore sulla validazione B: {best_name} (AUC {ranking.iloc[0]:.4f})")

    best_model = models[best_name].estimator.fit(X, data.y)
    joblib.dump(
        {"model": best_model, "features": reduction.kept_features},
        OUTPUT_DIR / "best_model.joblib",
    )

    print("[Fase 4] Spiegazione...")
    if best_name == "logistic_regression":
        # Per un modello lineare i coefficienti standardizzati SONO la spiegazione.
        # Un modello lineare e' monotono per costruzione: la direzione e' sempre
        # dichiarabile e coincide col segno del coefficiente.
        coefficients = best_model.steps[-1][1].coef_[0]
        importance = pd.DataFrame({
            "feature": reduction.kept_features,
            "importance": np.abs(coefficients),
            "direction": ["FL" if c > 0 else "REACTIVE" for c in coefficients],
            "direction_rho": np.sign(coefficients),
            "quintile_profile": "",
        }).sort_values("importance", ascending=False, ignore_index=True)
    else:
        importance = explain_with_shap(best_model, X, reduction.kept_features)
        _plot_shap_summary(best_model, X, reduction.kept_features, IMG_DIR / "shap_summary.png")
    importance.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)

    print("[Fase 4] Figure...")
    _plot_roc(predictions, IMG_DIR / "roc_curves.png")
    _plot_validation_gap(metrics, IMG_DIR / "validation_gap.png")
    _plot_shap_vs_univariate(importance, IMG_DIR / "shap_vs_univariate.png")

    summary = {}
    for model in models:
        optimistic = float(metrics_a[metrics_a.model == model]["auc_roc"].mean())
        conservative = float(metrics_b[metrics_b.model == model]["auc_roc"].mean())
        summary[model] = {
            "A_casuale": optimistic,
            "B_blocchi": conservative,
            "forbice": optimistic - conservative,
        }

    metadata = {
        "fase": "Fase 4 — Classificazione tabulare e spiegabilita'",
        "spec": "docs/superpowers/specs/2026-08-20-fase4-classificazione-design.md",
        "seed": SEED,
        "ambiente": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
        },
        "riduzione_ridondanze": {
            "soglia_rho": REDUNDANCY_THRESHOLD,
            "n_biomarcatori_iniziali": len(data.feature_names),
            "n_biomarcatori_tenuti": len(reduction.kept_features),
            "tenuti": reduction.kept_features,
        },
        "validazione": {
            "n_splits": N_SPLITS,
            "block_size_principale": MAIN_BLOCK_SIZE,
            "block_sizes_sensibilita": list(BLOCK_SIZES),
            "n_blocchi_principale": int(len(set(groups))),
        },
        "griglie": {name: spec.param_grid for name, spec in models.items()},
        "auc_roc_medio": summary,
        "modello_migliore": best_name,
        "nota": (
            "Il modello migliore e' scelto sulla validazione B (conservativa): "
            "sceglierlo su A premierebbe chi sfrutta meglio il leakage."
        ),
    }
    (OUTPUT_DIR / "classification_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[Fase 4] Completata. Artefatti in {OUTPUT_DIR} e {IMG_DIR}.")
    for model, values in summary.items():
        print(f"  {model:22s} A {values['A_casuale']:.4f}  B {values['B_blocchi']:.4f}  "
              f"forbice {values['forbice']:+.4f}")


if __name__ == "__main__":
    main()
