"""
===============================================================================
stain_robustness.py — La tessitura legge la cromatina o il vetrino?
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
===============================================================================
LA DOMANDA. L'ablazione per famiglie (Fase 4) mostra che cinque biomarcatori di
tessitura e intensita' eguagliano da soli tutti e 33, mentre i 28 morfometrici e
spaziali si fermano a 0.857 di AUC. Ma tessitura e intensita' del canale
ematossilina sono anche le grandezze piu' esposte alla variabilita' tecnica:
lotto di colorazione, spessore della sezione, resa dello scanner. Se il modello
stesse leggendo la firma del vetrino invece della cromatina, il risultato
principale della tesi poggerebbe su un artefatto.

IL METODO. Si perturba artificialmente la colorazione delle immagini grezze e si
rifa' girare l'INTERA pipeline — normalizzazione di Macenko compresa. La
perturbazione e' quella di Tellez et al. (2019): nello spazio delle
concentrazioni di ematossilina ed eosina, ogni canale viene alterato in modo
moltiplicativo E additivo,

    c' = alpha * c + beta,   alpha ~ U(1-s, 1+s),   beta ~ U(-s, +s) * media(c)

La componente additiva e' essenziale. La normalizzazione di Macenko riscala le
concentrazioni al 99esimo percentile della reference: una perturbazione solo
moltiplicativa verrebbe quindi riassorbita quasi per intero, e il test
misurerebbe la propria stessa inefficacia. La parte additiva sposta la forma
della distribuzione e sopravvive alla normalizzazione.

COSA SI MISURA.
  1. Quanto si spostano i biomarcatori, per famiglia, in unita' di IQR del
     dataset — cosi' grandezze con scale diverse sono confrontabili.
  2. Quanto si sposta la probabilita' predetta, e quante patch cambiano classe.
  3. Quanto cala l'AUC del modello sulle feature perturbate.

COME SI LEGGE. Se sotto perturbazione la tessitura si sposta molto piu' della
morfometria e le predizioni diventano instabili, quelle feature stavano leggendo
il vetrino. Se reggono, stavano leggendo la cromatina — e la conclusione della
tesi e' solida.

Esecuzione:
    python src/stain_robustness.py
===============================================================================
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import gui_core  # noqa: E402  (riusa la pipeline invece di reimplementarla)
from naming import CATEGORIES, short_label  # noqa: E402

BASE_DIR = _SRC_DIR.parent
RAW_DIR = BASE_DIR / "data" / "raw"
FASE1_DIR = BASE_DIR / "data" / "fase1_preprocessing"
FASE3_DIR = BASE_DIR / "data" / "fase3_features"
FASE4_DIR = BASE_DIR / "data" / "fase4_classification"
IMG_FASE4_DIR = BASE_DIR / "img" / "fase4"

SEED = 42
SAMPLE_PER_CLASS = 50
SIGMAS = (0.0, 0.10, 0.20, 0.30)
OPTICAL_DENSITY_MAX = 255

# Le due famiglie messe a confronto. La prima e' quella sospetta: descrive
# intensita' e tessitura del segnale, non la geometria dei nuclei.
TEXTURE_AND_INTENSITY = (
    "hchannel_mean", "hchannel_std",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "lbp_entropy",
)


def _prep():
    """Il modulo della Fase 1, che possiede l'implementazione di Macenko."""
    spec = importlib.util.spec_from_file_location(
        "stain_preprocessing", _SRC_DIR / "01_preprocessing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def perturb_stain(
    image_rgb: np.ndarray, sigma: float, rng: np.random.Generator, estimator=None
) -> np.ndarray:
    """
    Altera la colorazione di un'immagine H&E lasciandone intatta la geometria.

    Si scompone l'immagine nelle concentrazioni di ematossilina ed eosina con lo
    stesso stimatore usato dalla Fase 1, si perturbano le concentrazioni e si
    ricompone. I nuclei restano dove sono e della forma che hanno: cambia solo
    quanto e come sono colorati. E' precisamente la variabilita' che distingue
    due lotti di colorazione o due scanner.

    Args:
        image_rgb: immagine RGB uint8.
        sigma: intensita' della perturbazione. 0 restituisce l'immagine
            ricomposta senza alterazioni (utile come controllo: isola l'effetto
            del solo ciclo scomposizione/ricomposizione).
        rng: generatore, per rendere l'esperimento riproducibile.
        estimator: istanza di StainNormalizerMacenko da riusare fra chiamate.

    Returns:
        Immagine RGB uint8 della stessa forma.
    """
    if sigma < 0:
        raise ValueError(f"sigma non puo' essere negativo: {sigma}")

    if estimator is None:
        estimator = _prep().StainNormalizerMacenko()

    height, width, _ = image_rgb.shape
    flat = image_rgb.astype(np.float64).reshape((-1, 3))
    optical_density = -np.log10((flat + 1.0) / (OPTICAL_DENSITY_MAX + 1.0))

    stain_vectors = estimator._estimate_HE_vectors(optical_density)
    concentrations = np.linalg.lstsq(stain_vectors, optical_density.T, rcond=None)[0]

    if sigma > 0:
        for stain in range(concentrations.shape[0]):
            alpha = rng.uniform(1.0 - sigma, 1.0 + sigma)
            beta = rng.uniform(-sigma, sigma) * np.abs(concentrations[stain]).mean()
            concentrations[stain] = alpha * concentrations[stain] + beta

    perturbed_od = np.dot(stain_vectors, concentrations)
    perturbed = OPTICAL_DENSITY_MAX * (10 ** (-perturbed_od))
    return np.clip(perturbed.T, 0, 255).astype(np.uint8).reshape((height, width, 3))


def _sample_patches(rng: np.random.Generator) -> list[tuple[str, str]]:
    """Un campione stratificato di patch, per classe."""
    master = pd.read_csv(FASE3_DIR / "features_patches_master.csv")
    chosen = []
    for category in CATEGORIES:
        names = master.loc[master["category"] == category, "image_name"].tolist()
        picked = rng.choice(names, size=min(SAMPLE_PER_CLASS, len(names)), replace=False)
        chosen.extend((category, str(name)) for name in picked)
    return chosen


def run_experiment(sigmas=SIGMAS, seed: int = SEED) -> pd.DataFrame:
    """
    Rielabora ogni patch del campione a ciascun livello di perturbazione.

    Returns:
        DataFrame con una riga per (patch, sigma): i biomarcatori ricalcolati,
        la probabilita' predetta e la classe reale.
    """
    import cv2

    rng = np.random.default_rng(seed)
    patches = _sample_patches(rng)
    estimator = _prep().StainNormalizerMacenko()
    normalizer = gui_core.build_normalizer(
        gui_core.load_reference_image(FASE1_DIR, RAW_DIR)
    )
    classifier = gui_core.load_classifier(FASE4_DIR)

    records = []
    for index, (category, stem) in enumerate(patches, start=1):
        bgr = cv2.imread(str(RAW_DIR / category / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"immagine grezza mancante: {stem}")
        raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for sigma in sigmas:
            # Un generatore per patch e sigma: la perturbazione e' casuale ma
            # riproducibile, e non dipende dall'ordine di iterazione.
            local_rng = np.random.default_rng(abs(hash((stem, float(sigma)))) % (2**32))
            image = perturb_stain(raw, sigma, local_rng, estimator=estimator)
            result = gui_core.process_image(image, normalizer)

            row = {
                "image_name": stem,
                "category": category,
                "sigma": sigma,
                "target": 1 if category == CATEGORIES[0] else 0,
                "n_nuclei": len(result["nuclei"]),
            }
            row.update(result["features"])
            row["probability"] = (
                gui_core.predict_patch(classifier, result["features"])
                if result["nuclei"] else np.nan
            )
            records.append(row)

        if index % 20 == 0:
            print(f"  {index}/{len(patches)} patch elaborate")

    return pd.DataFrame(records)


def summarise_feature_shift(experiment: pd.DataFrame) -> pd.DataFrame:
    """
    Di quanto si sposta ogni biomarcatore, in unita' di IQR del dataset.

    Normalizzare per l'IQR rende confrontabili grandezze con scale diverse: uno
    spostamento di 1.0 significa che la perturbazione ha mosso il biomarcatore
    quanto l'intero scarto interquartile del dataset — enorme. Sotto 0.1 e'
    trascurabile.
    """
    master = pd.read_csv(FASE3_DIR / "features_patches_master.csv")
    features = [c for c in master.columns if c not in ("image_name", "category", "target")]
    iqr = (master[features].quantile(0.75) - master[features].quantile(0.25)).replace(0, np.nan)

    baseline = experiment[experiment["sigma"] == 0].set_index("image_name")
    records = []
    for sigma in sorted(experiment["sigma"].unique()):
        if sigma == 0:
            continue
        perturbed = experiment[experiment["sigma"] == sigma].set_index("image_name")
        common = baseline.index.intersection(perturbed.index)
        shift = (perturbed.loc[common, features] - baseline.loc[common, features]).abs() / iqr
        for feature in features:
            records.append({
                "sigma": sigma,
                "feature": feature,
                "famiglia": ("tessitura/intensita" if feature in TEXTURE_AND_INTENSITY
                             else "morfometria/spaziale"),
                "spostamento_mediano_IQR": float(shift[feature].median()),
            })
    return pd.DataFrame(records)


def summarise_prediction_stability(experiment: pd.DataFrame) -> pd.DataFrame:
    """Quanto si muove la predizione, e quante patch cambiano classe."""
    from sklearn.metrics import roc_auc_score

    baseline = experiment[experiment["sigma"] == 0].set_index("image_name")
    records = []
    for sigma in sorted(experiment["sigma"].unique()):
        subset = experiment[experiment["sigma"] == sigma].set_index("image_name")
        valid = subset.dropna(subset=["probability"])
        common = baseline.index.intersection(valid.index)

        delta = (valid.loc[common, "probability"] - baseline.loc[common, "probability"]).abs()
        flipped = (
            (valid.loc[common, "probability"] >= 0.5)
            != (baseline.loc[common, "probability"] >= 0.5)
        )
        records.append({
            "sigma": sigma,
            "auc": float(roc_auc_score(valid["target"], valid["probability"])),
            "delta_probabilita_mediana": float(delta.median()),
            "delta_probabilita_p90": float(delta.quantile(0.90)),
            "patch_che_cambiano_classe": int(flipped.sum()),
            "n_patch": int(len(common)),
        })
    return pd.DataFrame(records)


def _plot(shift: pd.DataFrame, stability: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    by_family = shift.groupby(["sigma", "famiglia"])["spostamento_mediano_IQR"].median().unstack()
    by_family.plot(kind="bar", ax=left, rot=0)
    left.set_xlabel("intensita' della perturbazione (sigma)")
    left.set_ylabel("spostamento mediano (unita' di IQR)")
    left.set_title("Quanto si spostano i biomarcatori")
    left.legend(title=None, fontsize=9)

    right.plot(stability["sigma"], stability["auc"], marker="o")
    right.set_xlabel("intensita' della perturbazione (sigma)")
    right.set_ylabel("AUC-ROC")
    right.set_title("Tiene la capacita' discriminante?")
    right.set_ylim(0.5, 1.0)
    right.grid(alpha=0.3)

    fig.suptitle("Robustezza alla variabilita' di colorazione")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FASE4_DIR.mkdir(parents=True, exist_ok=True)
    IMG_FASE4_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Robustezza] {SAMPLE_PER_CLASS} patch per classe, sigma {SIGMAS}...")
    experiment = run_experiment()
    experiment.to_csv(FASE4_DIR / "stain_robustness_raw.csv", index=False)

    shift = summarise_feature_shift(experiment)
    shift.to_csv(FASE4_DIR / "stain_robustness_feature_shift.csv", index=False)

    stability = summarise_prediction_stability(experiment)
    stability.to_csv(FASE4_DIR / "stain_robustness_stability.csv", index=False)

    _plot(shift, stability, IMG_FASE4_DIR / "stain_robustness.png")

    print("\nSpostamento mediano dei biomarcatori (unita' di IQR del dataset):")
    print(shift.groupby(["sigma", "famiglia"])["spostamento_mediano_IQR"].median()
          .unstack().round(3).to_string())
    print("\nStabilita' della predizione:")
    print(stability.round(4).to_string(index=False))
    print(f"\n[Robustezza] Completata. Artefatti in {FASE4_DIR} e {IMG_FASE4_DIR}.")


if __name__ == "__main__":
    main()
