"""
Test della Fase 4 — classificazione tabulare e spiegabilita' (src/04_classification.py).

Spec: docs/superpowers/specs/2026-08-20-fase4-classificazione-design.md

I test qui non inseguono la copertura: ciascuno presidia una proprieta' che, se
violata, renderebbe falso un numero della tesi. La piu' importante e' l'assenza
di sovrapposizione fra addestramento e test nella validazione a blocchi: e' la
proprieta' che giustifica l'intera decisione D1.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"


@pytest.fixture(scope="module")
def clf():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    spec = importlib.util.spec_from_file_location(
        "classification_under_test", SRC_DIR / "04_classification.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _names(prefix: str, indices) -> list[str]:
    return [f"{prefix}_examples ({i})" for i in indices]


# --------------------------------------------------------------------------
# Blocchi contigui — la sonda che sostituisce le etichette di caso assenti
# --------------------------------------------------------------------------
def test_contiguous_blocks_groups_neighbouring_indices_together(clf):
    names = _names("FL", range(1, 11))

    blocks = clf.contiguous_blocks(names, block_size=5)

    assert len(set(blocks)) == 2, "dieci patch con blocchi da cinque danno due gruppi"
    assert blocks[0] == blocks[4] and blocks[5] == blocks[9]
    assert blocks[4] != blocks[5], "il confine di blocco non e' stato rispettato"


def test_contiguous_blocks_follows_the_numeric_index_not_the_alphabetical_order(clf):
    """
    'FL_examples (10)' precede alfabeticamente 'FL_examples (9)': ordinare per
    stringa spezzerebbe i vicini proprio dove servono uniti.
    """
    names = _names("FL", [1, 2, 9, 10, 11])

    blocks = clf.contiguous_blocks(names, block_size=2)

    by_name = dict(zip(names, blocks))
    assert by_name["FL_examples (9)"] != by_name["FL_examples (1)"]
    assert by_name["FL_examples (10)"] == by_name["FL_examples (9)"]


def test_contiguous_blocks_never_mixes_the_two_categories(clf):
    """
    Un blocco a cavallo delle classi renderebbe impossibile lo split stratificato
    e mescolerebbe pazienti diversi per definizione.
    """
    names = _names("FL", range(1, 4)) + _names("REACTIVE", range(1, 4))

    blocks = clf.contiguous_blocks(names, block_size=100)

    assert blocks[0] == blocks[2], "le tre FL dovrebbero stare insieme"
    assert blocks[3] == blocks[5], "le tre REACTIVE dovrebbero stare insieme"
    assert blocks[2] != blocks[3], "FL e REACTIVE finite nello stesso blocco"


def test_contiguous_blocks_rejects_a_name_without_an_index(clf):
    with pytest.raises(ValueError, match="indice"):
        clf.contiguous_blocks(["patch_senza_numero"], block_size=5)


def test_contiguous_blocks_assigns_every_patch_exactly_one_block(clf):
    names = _names("FL", range(1, 301)) + _names("REACTIVE", range(1, 301))

    blocks = clf.contiguous_blocks(names, block_size=10)

    assert len(blocks) == 600
    assert not np.any(np.equal(blocks, None))
    # 300 patch per classe divise in blocchi da 10 -> 30 blocchi per classe
    assert len(set(blocks)) == 60


# --------------------------------------------------------------------------
# Caricamento della matrice dei biomarcatori
# --------------------------------------------------------------------------
MASTER_CSV = BASE_DIR / "data" / "fase3_features" / "features_patches_master.csv"


@pytest.fixture(scope="module")
def dataset(clf):
    if not MASTER_CSV.exists():
        pytest.skip("features_patches_master.csv non presente: eseguire la Fase 3.")
    return clf.load_feature_matrix(MASTER_CSV)


def test_load_feature_matrix_returns_the_full_balanced_dataset(dataset):
    data = dataset

    assert data.X.shape == (600, 47)
    assert sorted(np.bincount(data.y).tolist()) == [300, 300]


def test_load_feature_matrix_respects_the_phase3_feature_contract(clf, dataset):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "features_contract", SRC_DIR / "03_feature_extraction.py"
    )
    extraction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extraction)

    assert list(dataset.feature_names) == list(extraction.PATCH_FEATURE_COLUMNS)


def test_load_feature_matrix_refuses_missing_values(dataset):
    """
    La Fase 3 garantisce zero NaN. Se ne comparissero, qualcosa a monte e'
    cambiato: va indagato, non imputato in silenzio.
    """
    assert not np.isnan(dataset.X).any()


def test_load_feature_matrix_fails_loudly_on_a_missing_file(clf):
    with pytest.raises(FileNotFoundError):
        clf.load_feature_matrix(BASE_DIR / "data" / "non_esiste.csv")


# --------------------------------------------------------------------------
# LA guardia: nessuna sovrapposizione fra addestramento e test
# --------------------------------------------------------------------------
def test_block_validation_never_shares_a_block_between_train_and_test(clf, dataset):
    """
    E' la proprieta' che giustifica l'intera decisione D1. Se un blocco comparisse
    da entrambe le parti, la validazione "conservativa" starebbe misurando lo
    stesso leakage di quella casuale e la forbice pubblicata sarebbe priva di
    significato.
    """
    from sklearn.model_selection import GroupKFold

    groups = np.array(clf.contiguous_blocks(dataset.image_names, block_size=10))

    for train_idx, test_idx in GroupKFold(n_splits=5).split(dataset.X, dataset.y, groups):
        shared = set(groups[train_idx]) & set(groups[test_idx])
        assert not shared, f"blocchi presenti sia in addestramento sia in test: {shared}"


def test_random_validation_does_share_blocks_between_train_and_test(clf, dataset):
    """
    Il contrappunto del test precedente: lo split casuale MESCOLA i blocchi. Se
    non lo facesse, le due validazioni sarebbero la stessa cosa e confrontarle
    non direbbe nulla.
    """
    from sklearn.model_selection import StratifiedKFold

    groups = np.array(clf.contiguous_blocks(dataset.image_names, block_size=10))

    train_idx, test_idx = next(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(dataset.X, dataset.y)
    )
    assert set(groups[train_idx]) & set(groups[test_idx])


# --------------------------------------------------------------------------
# D3 — riduzione delle ridondanze, con criterio di leggibilita' clinica
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reduction(clf, dataset):
    return clf.reduce_redundant_features(dataset.X, dataset.feature_names, threshold=0.90)


def test_reduction_keeps_one_representative_per_group(reduction):
    groups = {}
    for row in reduction.assignments:
        groups.setdefault(row["group"], []).append(row)

    for group_id, members in groups.items():
        kept = [m for m in members if m["kept"]]
        assert len(kept) == 1, f"gruppo {group_id}: {len(kept)} rappresentanti invece di 1"
        assert kept[0]["representative"] == kept[0]["feature"]


def test_reduction_covers_every_original_biomarker(reduction, dataset):
    assert {row["feature"] for row in reduction.assignments} == set(dataset.feature_names)
    assert set(reduction.kept_features) <= set(dataset.feature_names)


def test_reduction_leaves_about_thirty_three_biomarkers(reduction):
    assert len(reduction.kept_features) == 33


def test_reduction_prefers_the_readable_variable_over_the_derived_one(reduction):
    """
    n_nuclei e nuclear_density_per_1000um2 sono la stessa grandezza (rho = 1.0):
    va tenuto il conteggio, che un patologo puo' nominare, non la densita'
    derivata dividendo per un'area costante.
    """
    assert "n_nuclei" in reduction.kept_features
    assert "nuclear_density_per_1000um2" not in reduction.kept_features


def test_reduction_prefers_the_mean_over_its_higher_order_moments(reduction):
    """
    Fra 'circolarita' media' e 'asimmetria della circolarita'' la prima si
    spiega a voce, la seconda no. Il criterio statistico avrebbe scelto l'altra.
    """
    assert "circularity_mean" in reduction.kept_features
    assert "circularity_skew" not in reduction.kept_features


def test_reduction_records_which_feature_replaces_each_discarded_one(reduction):
    """Le scartate non spariscono: si deve poter dire da chi sono rappresentate."""
    discarded = [row for row in reduction.assignments if not row["kept"]]

    assert discarded, "nessuna feature scartata: la riduzione non ha fatto nulla"
    for row in discarded:
        assert row["representative"] in reduction.kept_features
        assert row["representative"] != row["feature"]


def test_reduction_at_the_strictest_threshold_only_merges_identical_variables(clf, dataset):
    """
    Soglia 1.0: si fondono solo variabili con correlazione esattamente 1. Su
    questi dati ne esiste una coppia sola, n_nuclei e la densita' nucleare, che
    e' il conteggio diviso per un'area costante (Fase 3, §3.3).
    """
    full = clf.reduce_redundant_features(dataset.X, dataset.feature_names, threshold=1.0)

    assert len(full.kept_features) == len(dataset.feature_names) - 1
    assert "nuclear_density_per_1000um2" not in full.kept_features
    assert "n_nuclei" in full.kept_features


# --------------------------------------------------------------------------
# D2 / D4 — modelli e valutazione annidata
# --------------------------------------------------------------------------
def test_build_models_provides_the_three_declared_models(clf):
    models = clf.build_models(seed=42)

    assert set(models) == {"logistic_regression", "random_forest", "xgboost"}


def test_only_the_linear_model_standardises_and_it_does_so_inside_the_pipeline(clf):
    """
    Adattare lo scaler fuori dallo split e' leakage, della stessa famiglia di D1.
    Gli alberi non ne hanno bisogno (Fase 3, §5): scalarli sarebbe rumore.
    """
    models = clf.build_models(seed=42)

    steps = {name: [s for s, _ in spec.estimator.steps] for name, spec in models.items()}
    assert "scaler" in steps["logistic_regression"]
    assert "scaler" not in steps["random_forest"]
    assert "scaler" not in steps["xgboost"]


@pytest.fixture(scope="module")
def small_evaluation(clf, dataset):
    """Valutazione su un sottoinsieme, per restare rapidi: qui conta la forma."""
    from sklearn.model_selection import StratifiedKFold

    subset = np.r_[0:60, 300:360]
    models = {"logistic_regression": clf.build_models(seed=42)["logistic_regression"]}
    return clf.evaluate(
        X=dataset.X[subset],
        y=dataset.y[subset],
        models=models,
        splitter=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        validation="A_casuale",
        seed=42,
    )


def test_evaluate_returns_one_row_per_model_and_fold(small_evaluation):
    assert len(small_evaluation) == 3
    assert set(small_evaluation["fold"]) == {0, 1, 2}


def test_evaluate_reports_the_metrics_the_spec_requires(small_evaluation):
    expected = {
        "validation", "model", "fold", "auc_roc", "balanced_accuracy",
        "sensitivity", "specificity", "n_train", "n_test", "best_params",
    }
    assert expected <= set(small_evaluation.columns)


def test_evaluate_produces_metrics_in_the_valid_range(small_evaluation):
    for column in ("auc_roc", "balanced_accuracy", "sensitivity", "specificity"):
        values = small_evaluation[column]
        assert values.between(0.0, 1.0).all(), f"{column} fuori da [0, 1]"


def test_evaluate_is_reproducible_with_the_same_seed(clf, dataset):
    from sklearn.model_selection import StratifiedKFold

    subset = np.r_[0:60, 300:360]
    models = {"logistic_regression": clf.build_models(seed=42)["logistic_regression"]}

    def run():
        return clf.evaluate(
            X=dataset.X[subset], y=dataset.y[subset], models=models,
            splitter=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            validation="A_casuale", seed=42,
        )["auc_roc"].tolist()

    assert run() == run()


# --------------------------------------------------------------------------
# Sensibilita' alla dimensione del blocco (D1) e spiegazione SHAP (D6)
# --------------------------------------------------------------------------
def test_block_size_sensitivity_reports_every_requested_size(clf, dataset):
    subset = np.r_[0:60, 300:360]
    models = {"logistic_regression": clf.build_models(seed=42)["logistic_regression"]}

    # Blocchi da 5 e 10 su 120 patch: con blocchi piu' grandi i gruppi
    # diventerebbero cosi' pochi da produrre pieghe monoclasse, e la guardia di
    # evaluate() interverrebbe — correttamente, ma qui si sta collaudando
    # un'altra cosa.
    table = clf.block_size_sensitivity(
        X=dataset.X[subset], y=dataset.y[subset],
        image_names=[dataset.image_names[i] for i in subset],
        models=models, block_sizes=(5, 10), seed=42, n_splits=3,
    )

    assert set(table["block_size"]) == {5, 10}
    assert table["n_blocks"].loc[table.block_size == 5].iloc[0] > \
           table["n_blocks"].loc[table.block_size == 10].iloc[0], \
           "blocchi piu' grandi devono dare meno gruppi"


def test_shap_importance_covers_every_feature_once(clf, dataset):
    reduction = clf.reduce_redundant_features(dataset.X, dataset.feature_names, 0.90)
    columns = [dataset.feature_names.index(f) for f in reduction.kept_features]
    X = dataset.X[:, columns]

    spec = clf.build_models(seed=42)["random_forest"]
    model = spec.estimator.fit(X, dataset.y)

    importance = clf.explain_with_shap(model, X, reduction.kept_features)

    assert list(importance["feature"]) == sorted(
        importance["feature"], key=lambda f: -importance.set_index("feature").loc[f, "importance"]
    )
    assert set(importance["feature"]) == set(reduction.kept_features)
    assert (importance["importance"] >= 0).all()


def test_shap_ranking_agrees_with_the_phase3_univariate_analysis(clf, dataset):
    """
    Coerenza attesa dalla spec: lbp_entropy e' il biomarcatore piu' discriminante
    secondo i test univariati della Fase 3 (p FDR = 3.2e-51). Se sparisse dalle
    prime posizioni della classifica SHAP, il sospetto sarebbe un errore prima
    che una scoperta.
    """
    reduction = clf.reduce_redundant_features(dataset.X, dataset.feature_names, 0.90)
    columns = [dataset.feature_names.index(f) for f in reduction.kept_features]
    X = dataset.X[:, columns]

    model = clf.build_models(seed=42)["random_forest"].estimator.fit(X, dataset.y)
    importance = clf.explain_with_shap(model, X, reduction.kept_features)

    top_five = importance["feature"].head(5).tolist()
    assert "lbp_entropy" in top_five, f"lbp_entropy fuori dai primi cinque: {top_five}"


def test_evaluate_refuses_a_test_fold_with_a_single_class(clf, dataset):
    """
    Con blocchi che non mescolano mai le classi, uno splitter puo' produrre una
    piega di test monoclasse: l'AUC diventa indefinita e, mediata in silenzio,
    produrrebbe un numero falso. Deve fallire rumorosamente.
    """
    from sklearn.model_selection import GroupKFold

    only_fl = np.arange(0, 40)
    models = {"logistic_regression": clf.build_models(seed=42)["logistic_regression"]}
    groups = np.array(clf.contiguous_blocks(
        [dataset.image_names[i] for i in only_fl], block_size=10))

    with pytest.raises(ValueError, match="una sola classe"):
        clf.evaluate(
            X=dataset.X[only_fl], y=dataset.y[only_fl], models=models,
            splitter=GroupKFold(n_splits=2), validation="degenere",
            seed=42, groups=groups,
        )


def test_shap_declares_a_direction_only_when_the_effect_is_monotone(clf, dataset):
    """
    Una correlazione lineare fra valore e contributo e' priva di senso per un
    effetto a U. solidity_mean e' proprio questo: medie di classe quasi identiche
    (Mann-Whitney p = 0.106) ma dispersioni molto diverse (Levene p = 3.0e-06),
    per cui valori estremi in ENTRAMBE le direzioni indicano linfoma. Dichiararne
    una direzione metterebbe un'affermazione falsa nella tesi.
    """
    reduction = clf.reduce_redundant_features(dataset.X, dataset.feature_names, 0.90)
    columns = [dataset.feature_names.index(f) for f in reduction.kept_features]
    X = dataset.X[:, columns]

    model = clf.build_models(seed=42)["xgboost"].estimator.fit(X, dataset.y)
    importance = clf.explain_with_shap(model, X, reduction.kept_features).set_index("feature")

    assert importance.loc["lbp_entropy", "direction"] == "REACTIVE", \
        "entropia LBP alta indica tessuto reattivo (Fase 3, §3.2)"
    assert importance.loc["solidity_mean", "direction"] == "non monotona"
    assert set(importance["direction"]) <= {"FL", "REACTIVE", "non monotona"}


# --------------------------------------------------------------------------
# Contributo per famiglia di biomarcatori
# --------------------------------------------------------------------------
def test_contribution_reports_every_family(clf, dataset):
    from sklearn.model_selection import GroupKFold  # noqa: F401

    subset = np.r_[0:60, 300:360]
    names = [dataset.image_names[i] for i in subset]
    groups = np.array(clf.contiguous_blocks(names, block_size=5))
    models = {"logistic_regression": clf.build_models(seed=42)["logistic_regression"]}

    table = clf.feature_family_contribution(
        dataset.X[subset], dataset.y[subset], list(dataset.feature_names),
        models, groups, seed=42, n_splits=3,
    )

    assert set(table["sottoinsieme"]) == {
        "tutte", "senza intensita'", "senza tessitura",
        "solo morfometria e spaziale", "solo tessitura e intensita'",
    }
    assert table["auc_roc"].between(0.0, 1.0).all()

    summary = clf.summarise_contribution(table)
    assert len(summary) == 5, "un modello per cinque sottoinsiemi"
    assert summary["auc_roc_medio"].between(0.0, 1.0).all()


def test_contribution_keeps_the_per_fold_scores(clf, dataset):
    """
    Le AUC per piega devono sopravvivere all'aggregazione.

    Senza di esse il confronto appaiato fra sottoinsiemi non e' ricalcolabile
    dagli artefatti salvati, ed e' il solo confronto lecito: i sottoinsiemi
    condividono le stesse pieghe.
    """
    subset = np.r_[0:60, 300:360]
    names = [dataset.image_names[i] for i in subset]
    groups = np.array(clf.contiguous_blocks(names, block_size=5))
    models = {"logistic_regression": clf.build_models(seed=42)["logistic_regression"]}

    table = clf.feature_family_contribution(
        dataset.X[subset], dataset.y[subset], list(dataset.feature_names),
        models, groups, seed=42, n_splits=3,
    )

    assert {"fold", "auc_roc"} <= set(table.columns)
    assert len(table) == 5 * 3, "cinque sottoinsiemi per tre pieghe"


def test_contribution_uses_the_same_folds_for_every_subset(clf, dataset):
    """
    L'appaiamento regge solo se la piega k contiene le stesse patch in ogni
    sottoinsieme. GroupKFold e' deterministico e i gruppi non cambiano, quindi
    deve essere cosi': se smettesse di esserlo, un test appaiato confronterebbe
    stime calcolate su insiemi diversi.
    """
    subset = np.r_[0:60, 300:360]
    names = [dataset.image_names[i] for i in subset]
    groups = np.array(clf.contiguous_blocks(names, block_size=5))
    models = {"logistic_regression": clf.build_models(seed=42)["logistic_regression"]}

    table = clf.feature_family_contribution(
        dataset.X[subset], dataset.y[subset], list(dataset.feature_names),
        models, groups, seed=42, n_splits=3,
    )

    sizes = table.pivot_table(index="fold", columns="sottoinsieme", values="n_test")
    assert (sizes.nunique(axis=1) == 1).all(), (
        "la stessa piega ha numerosita' diverse fra sottoinsiemi: non sono appaiati"
    )


def test_fold_metrics_panel_is_internally_consistent(clf):
    """
    Le identita' fra metriche devono valere esattamente, non circa.

    FNR = 1 - sensibilita', FPR = 1 - specificita', e F1 e' la media armonica di
    precisione e richiamo. Sono ridondanze volute: servono alla lettura clinica.
    Se una di esse smettesse di valere, il pannello conterrebbe due numeri che
    dicono cose diverse sotto nomi che promettono la stessa.
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1])
    y_pred = (y_prob >= 0.5).astype(int)

    m = clf._fold_metrics(y_true, y_prob, y_pred)

    assert m["sensitivity"] == pytest.approx(3 / 4)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["false_negative_rate"] == pytest.approx(1 - m["sensitivity"])
    assert m["false_positive_rate"] == pytest.approx(1 - m["specificity"])
    harmonic = 2 * m["precision"] * m["sensitivity"] / (m["precision"] + m["sensitivity"])
    assert m["f1"] == pytest.approx(harmonic)


def test_accuracy_and_balanced_accuracy_agree_on_balanced_folds(clf):
    """
    Con classi bilanciate le due coincidono, e il report ne cita una sola.

    Se le pieghe si sbilanciassero smetterebbero di coincidere: il test fissa
    l'ipotesi sotto cui l'equivalenza vale, cosi' che rompendola qualcosa lo
    segnali.
    """
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1])

    m = clf._fold_metrics(y_true, y_prob, (y_prob >= 0.5).astype(int))

    assert m["accuracy"] == pytest.approx(m["balanced_accuracy"])


def test_paired_tests_never_claim_unreachable_significance(clf, dataset):
    """
    Con poche pieghe il Wilcoxon appaiato non puo' scendere sotto una soglia.

    Con k pieghe le combinazioni di segno sono 2^k, quindi il p minimo a due code
    vale 2/2^k: con 5 pieghe e' 0.0625, sopra 0.05. Il test presidia che nessun
    valore riportato scenda sotto quel limite, perche' sarebbe un errore di
    calcolo, e che il limite sia dichiarato nella tabella accanto al p.
    """
    subset = np.r_[0:60, 300:360]
    names = [dataset.image_names[i] for i in subset]
    groups = np.array(clf.contiguous_blocks(names, block_size=5))
    models = {"logistic_regression": clf.build_models(seed=42)["logistic_regression"]}

    per_fold = clf.feature_family_contribution(
        dataset.X[subset], dataset.y[subset], list(dataset.feature_names),
        models, groups, seed=42, n_splits=3,
    )
    paired = clf.paired_family_tests(per_fold)

    assert len(paired) == len(clf.FAMILY_COMPARISONS)
    assert (paired["n_pieghe"] == 3).all()
    assert (paired["p_minimo_ottenibile"] == 0.25).all(), "2 / 2^3"
    assert (paired["p_wilcoxon"] >= paired["p_minimo_ottenibile"]).all()
    assert (paired["vittorie_a"] <= paired["n_pieghe"]).all()


def test_contribution_subsets_are_complementary(clf, dataset):
    """
    'solo morfometria' e 'solo tessitura' devono partizionare le feature: se si
    sovrapponessero o lasciassero fuori qualcosa, il confronto fra famiglie non
    direbbe cio' che sembra dire.
    """
    names = list(dataset.feature_names)
    tessitura = [f for f in names if f in clf.TEXTURE_AND_INTENSITY]
    morfometria = [f for f in names if f not in clf.TEXTURE_AND_INTENSITY]

    assert set(tessitura) & set(morfometria) == set()
    assert set(tessitura) | set(morfometria) == set(names)
    assert len(tessitura) == 6, "le feature di tessitura e intensita' dovrebbero essere sei"
