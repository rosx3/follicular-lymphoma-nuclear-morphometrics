# Fase 3 — Implementation Plan

> **Stato: APPROVATO NELLE DECISIONI, NON ANCORA IN ESECUZIONE.**
> Le decisioni D1–D7 (Sezione 2) sono state approvate il 19 agosto 2026.
> Nessun task della Sezione 4 è stato avviato.

**Obiettivo:** completare l'estrazione dei 47 biomarcatori citomorfometrici, micro-spaziali e di tessitura dalle 600 patch, eseguirla sull'intero dataset e quantificare la separabilità statistica FL vs REACTIVE.

**Architettura:** i biomarcatori vengono calcolati per singolo nucleo dalle maschere d'istanza della Fase 2 (`regionprops`), aggregati per patch con 4 statistiche (mean/std/skew/cv), affiancati dalle distanze k-NN sui centroidi in µm e dai descrittori di tessitura cromatinica calcolati sull'H-channel della Fase 1 ristretto ai pixel nucleari. L'output è una matrice tabulare 600 × 50 che alimenta la Fase 4 (ML tabulare + SHAP).

**Stack:** Python 3.14.3, numpy 2.4.3, scipy 1.17.1 (`KDTree`, `stats`, `false_discovery_control`), scikit-image 0.26.0 (`graycomatrix`, `graycoprops`, `local_binary_pattern`, `regionprops`), pandas 2.3.3 (sola analisi statistica), matplotlib 3.10.8, pytest 9.1.1.

**Spec di riferimento:**
- `reports/fase3_report.md` — set definitivo dei biomarcatori (§1–2) e motivazione delle esclusioni (§1.1, §7)
- `nuovaproposta_tesi_follicoli_linfatici.docx` — proposta di tesi approvata (workflow metodologico, punto 4)
- `reports/fase1_report.md`, `reports/fase2_report.md` — caratteristiche degli input

---

## 1. Stato di partenza

| Componente | Stato |
|---|---|
| `03_feature_extraction.py` STEP 1 — morfometria per nucleo | ✅ implementato |
| `03_feature_extraction.py` STEP 2 — aggregazione per patch (32 col) | ✅ implementato |
| `03_feature_extraction.py` STEP 3 — k-NN | ❌ `NotImplementedError` |
| `03_feature_extraction.py` STEP 4 — tessitura | ❌ `NotImplementedError` |
| `run_fase3` — cablaggio STEP 3/4 e metadata JSON | ❌ assenti |
| `data/fase3_features/` | ❌ mai generata |
| `reports/fase3_report.md` §3 (risultati) e §4 (figure) | ❌ vuote |
| Feature attualmente prodotte | 37 / 47 |

**Prerequisiti già chiusi** (audit di apertura Fase 3, sezione di fix completata il 19 agosto 2026):

| ID | Problema | Esito |
|---|---|---|
| B1 | Risoluzione percorsi incoerente fra le fasi; anteprima saltata in silenzio | ✅ `src/naming.py` + `iter_patch_inputs` / `iter_h_channel_inputs`, input mancante = `FileNotFoundError` |
| B2 | Tre convenzioni di categoria conviventi | ✅ canoniche ovunque, incl. `centroids_all.csv` e `split_gt_patches` |
| B3 | Docstring modulo 03 dichiarava 51 feature | ✅ allineata a 47 + 3 |
| — | Overlay Fase 2 saltato in silenzio | ✅ ora solleva |
| — | `requirements.txt` disallineato dall'ambiente reale | ✅ pin esatti verificati |
| — | `run_pipeline.py --fase 1` in crash (`fit()` riceveva un path) | ✅ corretto |

Verifica: **39 test** verdi (`python -m pytest tests/ -q`), di cui 5 di integrazione che eseguono Fase 1 → 2 → 3 su un mini-dataset temporaneo.

---

## 2. Decisioni metodologiche approvate

Queste decisioni chiudono le domande aperte poste prima dell'avvio. Sono vincolanti per tutti i task della Sezione 4.

### D1 — Patch con nuclei insufficienti: `NaN`, non `0.0`
*(era Q1)*

Con `n_nuclei < 2` la distanza al primo vicino non è definita; con `n_nuclei < 4` non lo è quella ai tre vicini. In questi casi le colonne k-NN valgono `NaN`.

**Perché:** `0.0` è un valore falso — verrebbe letto dal modello come "nuclei perfettamente sovrapposti", cioè il massimo di densità, esattamente l'opposto della realtà (patch quasi vuota). `NaN` è onesto e forza una scelta esplicita di imputazione in Fase 4.

**Conseguenza operativa:** il report §3.1 deve riportare quante patch sono affette, per categoria. Se il numero è non trascurabile va discusso come limite del dataset.

### D2 — Tessitura calcolata sui soli pixel nucleari
*(era Q2)*

GLCM, LBP e le statistiche di intensità dell'H-channel si calcolano **solo sui pixel appartenenti ai nuclei**, usando le maschere d'istanza della Fase 2 come maschera binaria.

**Perché:** il titolo della tesi e il report §2.6 parlano di *tessitura cromatinica*. Calcolarla sull'intera patch misurerebbe anche stroma, spazio inter-nucleare e sfondo, diluendo il segnale che si vuole quantificare.

**Conseguenza operativa:** la firma di `extract_texture_features` cambia — richiede anche la maschera. La scelta va dichiarata esplicitamente nel report §2.6, perché cambia l'interpretazione clinica delle 6 colonne.

### D3 — GLCM quantizzata a 64 livelli di grigio
*(era Q3)*

`GLCM_LEVELS = 64`, distanza 1 px, 4 angoli (0°, 45°, 90°, 135°) mediati.

**Perché:** è lo standard nella letteratura Haralick. Con 256 livelli la matrice di co-occorrenza è sparsa e le stime diventano instabili sul numero di pixel di una patch 224×224; 64 livelli danno stime più robuste e un calcolo circa 50× più veloce.

### D4 — Correzione per test multipli con Benjamini–Hochberg
*(era Q4)*

Nell'analisi di separabilità si riportano p-value grezzo **e** corretto FDR affiancati.

**Perché:** testando 47 feature simultaneamente a α=0.05 ci si attendono ~2 falsi positivi per puro caso. Senza correzione, una feature "significativa" non è distinguibile dal rumore.

### D5 — `pandas` limitato all'analisi statistica
*(era Q5)*

L'estrazione (Task 1–5) resta su `csv` della standard library; l'analisi e le figure (Task 6–7) usano pandas.

**Perché:** non riscrivere codice di estrazione già funzionante e testato solo per uniformità, ma non reimplementare a mano group-by e statistiche descrittive.

### D6 — Categorie canoniche
*(era Q6 — già implementata)*

`follicular_lymphoma` / `reactive_tissue`, come da `reports/fase3_report.md` §2.1, con `target` = 1 / 0. Risolta nella sezione di fix; `normalize_category()` in `src/naming.py` accetta le varianti storiche.

### D7 — Delaunay, MST, k=5 e CIE-LAB restano esclusi
*(era Q7)*

**Decisione: si mantiene l'esclusione**, documentandola nel report in modo da poterla riportare nella tesi.

**Contesto da documentare:** la proposta di tesi approvata richiede esplicitamente al punto 4 del workflow metodologico la triangolazione di Delaunay, il Minimum Spanning Tree, il k-NN con k=5 e i momenti cromatici CIE-LAB. La loro esclusione è quindi una **divergenza consapevole rispetto al documento approvato**, non un'omissione, e come tale va dichiarata.

**Motivazioni** (già in `reports/fase3_report.md` §1.1 e §7):
- Delaunay/MST: boundary effects critici su patch da 51.5 µm — il grafo viene troncato ai quattro bordi e nuclei biologicamente adiacenti ma su patch diverse risultano disconnessi. Sono strumenti validi alla scala della WSI (ordine dei mm), non della micro-patch.
- k=5: ridondante con k=3 alla scala micro-locale del dataset.
- CIE-LAB: ridondanti dopo la normalizzazione di Macenko, il cui scopo è proprio portare tutte le patch nello stesso spazio cromatico.

**Analisi esplorativa rinviata:** se resterà tempo, si calcoleranno comunque queste feature come analisi esplorativa, mostrando numericamente la loro correlazione con le k-NN e il loro potere discriminante, per sostituire l'argomento teorico con evidenza empirica. **Non fa parte dello scope corrente** e non compare fra i task della Sezione 4.

---

## 3. Vincoli globali

Valgono per ogni task; non vanno ripetuti nei singoli task ma sono parte implicita dei loro requisiti.

- **Calibrazione spaziale:** 1 px = 0.23 µm; area pixel = 0.0529 µm²; patch 224×224 px = 2654.31 µm². Costanti già in `src/03_feature_extraction.py`, non ridefinirle.
- **Unità fisiche:** ogni feature dimensionale è espressa in µm o µm², mai in pixel, nel CSV per patch. Le colonne in pixel restano solo nel CSV per singolo nucleo, a fini di audit.
- **Contratto di output:** `features_patches_master.csv` = 600 righe × 50 colonne (47 feature + `image_name`, `category`, `target`).
- **Categorie:** solo `follicular_lymphoma` / `reactive_tissue` (D6). Usare `src/naming.py`, mai stringhe letterali.
- **Determinismo:** nessun campionamento casuale nell'estrazione. Dove serve un seed (figure, sottocampionamenti), fissarlo e registrarlo nel metadata JSON.
- **Fallimento rumoroso:** un input mancante o un valore non calcolabile non deve produrre una riga assente o un valore plausibile ma falso. Vale il principio già applicato in B1.
- **Dipendenze:** nessuna nuova libreria oltre a quelle in `requirements.txt`. In particolare **niente `statsmodels`**: per l'FDR si usa `scipy.stats.false_discovery_control`.
- **Test:** ogni task si chiude con `python -m pytest tests/ -q` interamente verde.

---

## 4. Task

Ordine vincolante: il Task 3 dipende da 1 e 2; il Task 5 da 3 e 4; i Task 6–8 dal 5.

**Nota sul livello di dettaglio.** I Task 1–3 riportano il codice completo: sono il cuore dell'estrazione, la loro correttezza è verificabile in anticipo con test deterministici, e un errore lì si propaga a tutto il resto. I Task 4–8 riportano specifiche precise ma non codice riga per riga: dipendono da dati che al momento della stesura non esistono ancora — la scelta fra t-test e Mann–Whitney, per esempio, si decide sulla normalità osservata, non a priori. Fissarne il codice adesso sarebbe una precisione finta.

### Task 1 — STEP 3: distanze micro-spaziali k-NN

**File:**
- Modifica: `src/03_feature_extraction.py` (`compute_knn_spatial_features`, oggi `NotImplementedError`)
- Test: `tests/test_feature_knn.py` (nuovo)

**Interfacce prodotte:**
```python
KNN_NEIGHBOR_COUNTS: tuple[int, ...] = (1, 3)

def compute_knn_spatial_features(nuclei_list: list[dict]) -> dict:
    """Restituisce 4 chiavi:
    knn1_dist_mean_um, knn1_dist_std_um, knn3_dist_mean_um, knn3_dist_std_um
    """
```

**Definizione operativa** (da riportare nel report §2.5): per ogni nucleo si calcola la media delle distanze ai suoi *k* vicini più prossimi; `knnK_dist_mean_um` è la media di questo valore su tutti i nuclei della patch, `knnK_dist_std_um` la sua deviazione standard. Le distanze sono euclidee sui centroidi in µm (`centroid_x_um`, `centroid_y_um`). Per k=1 il valore per nucleo coincide con la distanza al primo vicino.

**Casi non definiti** (D1): servono almeno `k+1` nuclei. Con meno, le due colonne di quel k valgono `float("nan")`.

- [ ] **Step 1: scrivere i test che falliscono**

```python
# tests/test_feature_knn.py
import math
import importlib.util
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture(scope="module")
def features():
    spec = importlib.util.spec_from_file_location(
        "mod_features", SRC_DIR / "03_feature_extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nuclei(*coords):
    return [{"centroid_x_um": x, "centroid_y_um": y} for x, y in coords]


UNIT_SQUARE = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


def test_knn1_on_a_unit_square_is_the_side_length(features):
    result = features.compute_knn_spatial_features(_nuclei(*UNIT_SQUARE))

    assert result["knn1_dist_mean_um"] == pytest.approx(1.0)
    assert result["knn1_dist_std_um"] == pytest.approx(0.0)


def test_knn3_on_a_unit_square_averages_two_sides_and_one_diagonal(features):
    expected = (1.0 + 1.0 + math.sqrt(2.0)) / 3.0

    result = features.compute_knn_spatial_features(_nuclei(*UNIT_SQUARE))

    assert result["knn3_dist_mean_um"] == pytest.approx(expected)
    assert result["knn3_dist_std_um"] == pytest.approx(0.0)


def test_knn3_is_nan_with_only_three_nuclei_but_knn1_is_defined(features):
    result = features.compute_knn_spatial_features(_nuclei((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    assert result["knn1_dist_mean_um"] == pytest.approx(1.0)
    assert math.isnan(result["knn3_dist_mean_um"])
    assert math.isnan(result["knn3_dist_std_um"])


@pytest.mark.parametrize("coords", [(), ((0.0, 0.0),)])
def test_every_knn_column_is_nan_when_there_are_fewer_than_two_nuclei(features, coords):
    result = features.compute_knn_spatial_features(_nuclei(*coords))

    assert all(math.isnan(v) for v in result.values())


def test_returns_exactly_the_four_documented_columns(features):
    result = features.compute_knn_spatial_features(_nuclei(*UNIT_SQUARE))

    assert set(result) == {
        "knn1_dist_mean_um", "knn1_dist_std_um",
        "knn3_dist_mean_um", "knn3_dist_std_um",
    }


def test_distances_scale_with_the_coordinates(features):
    """Controllo dimensionale: raddoppiando le coordinate raddoppiano le distanze."""
    doubled = _nuclei(*[(2 * x, 2 * y) for x, y in UNIT_SQUARE])

    result = features.compute_knn_spatial_features(doubled)

    assert result["knn1_dist_mean_um"] == pytest.approx(2.0)
```

- [ ] **Step 2: eseguire i test e verificare che falliscano**

Run: `python -m pytest tests/test_feature_knn.py -q`
Atteso: FAIL con `NotImplementedError: STEP 3 — da implementare nella prossima iterazione`.

- [ ] **Step 3: implementare**

```python
from scipy.spatial import KDTree

KNN_NEIGHBOR_COUNTS: tuple[int, ...] = (1, 3)


def compute_knn_spatial_features(nuclei_list: list[dict]) -> dict:
    """
    Distanze micro-spaziali ai k vicini più prossimi (k = 1, 3), in µm.

    Per ogni nucleo si calcola la media delle distanze ai suoi k vicini più
    prossimi; le colonne restituite sono media e deviazione standard di questo
    valore sull'intera patch.

    Sostituisce Delaunay/MST come proxy di micro-architettura del packing
    nucleare (vedi reports/fase3_report.md §1.1 e §7, decisione D7 del piano).

    Con meno di k+1 nuclei la statistica non è definita e vale NaN: uno zero
    verrebbe interpretato dal modello come densità massima, cioè l'opposto
    della situazione reale (decisione D1 del piano).
    """
    nan = float("nan")
    result = {}
    for k in KNN_NEIGHBOR_COUNTS:
        result[f"knn{k}_dist_mean_um"] = nan
        result[f"knn{k}_dist_std_um"] = nan

    n_nuclei = len(nuclei_list)
    if n_nuclei < 2:
        return result

    coords = np.array(
        [[n["centroid_x_um"], n["centroid_y_um"]] for n in nuclei_list],
        dtype=np.float64,
    )
    tree = KDTree(coords)

    for k in KNN_NEIGHBOR_COUNTS:
        if n_nuclei < k + 1:
            continue
        # k+1 perché il primo vicino restituito è il nucleo stesso (distanza 0)
        distances, _ = tree.query(coords, k=k + 1)
        neighbor_distances = np.atleast_2d(distances)[:, 1:]
        per_nucleus = neighbor_distances.mean(axis=1)
        result[f"knn{k}_dist_mean_um"] = round(float(per_nucleus.mean()), 4)
        result[f"knn{k}_dist_std_um"] = round(float(per_nucleus.std()), 4)

    return result
```

- [ ] **Step 4: eseguire i test e verificare che passino**

Run: `python -m pytest tests/ -q`
Atteso: PASS, suite interamente verde.

- [ ] **Step 5: commit**

```bash
git add src/03_feature_extraction.py tests/test_feature_knn.py
git commit -m "feat(fase3): distanze micro-spaziali k-NN (k=1,3) con NaN sui casi non definiti"
```

---

### Task 2 — STEP 4: tessitura cromatinica sui pixel nucleari

**File:**
- Modifica: `src/03_feature_extraction.py` (`extract_texture_features`, oggi `NotImplementedError`)
- Test: `tests/test_feature_texture.py` (nuovo)

**Interfacce prodotte:**
```python
GLCM_LEVELS = 64
GLCM_DISTANCES = (1,)
GLCM_ANGLES_DEG = (0, 45, 90, 135)
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"

def extract_texture_features(
    h_channel_patch: np.ndarray, instance_mask: np.ndarray
) -> dict:
    """Restituisce 6 chiavi: glcm_contrast, glcm_homogeneity, glcm_energy,
    lbp_entropy, hchannel_mean, hchannel_std
    """
```

**Nota sulla firma:** rispetto allo scheletro attuale la funzione riceve anche la maschera d'istanza, perché la tessitura si calcola sui soli pixel nucleari (D2).

**Come si maschera una GLCM.** `graycomatrix` non accetta maschere. Si procede così: l'H-channel viene quantizzato su `GLCM_LEVELS - 1` = 63 livelli mappati su 1..63, e i pixel di sfondo vengono posti a 0. Si calcola la GLCM con `levels=64`, poi si **scartano riga 0 e colonna 0** — cioè tutte le coppie che coinvolgono almeno un pixel di sfondo. Non serve rinormalizzare a mano: `graycoprops` normalizza internamente ogni matrice (`P /= glcm_sums`), verificato sul sorgente di scikit-image 0.26.0.

**Perché lo scarto della riga/colonna 0 è lecito.** Dopo lo slicing gli indici di livello si spostano tutti di 1. `graycoprops` costruisce i pesi con `I, J = np.ogrid[0:num_level, 0:num_level]` e per contrasto e omogeneità li usa solo nella forma `(I - J)`: uno shift costante su entrambi si annulla, quindi i valori restano corretti. L'energia (ASM) non dipende affatto dagli indici. Le tre proprietà scelte sono esattamente quelle invarianti a questa operazione — `glcm_correlation`, che invece usa medie e indici assoluti e sarebbe distorta dallo slicing, è già esclusa dal set per altra ragione (report §1.1).

**Attenzione a un fallimento silenzioso:** se dopo lo scarto la GLCM è tutta a zeri (nessuna coppia di pixel nucleari adiacenti, es. nuclei ridotti a pixel isolati), `graycoprops` pone il denominatore a 1 e restituisce zeri — valori plausibili ma privi di significato. Il codice sotto intercetta il caso e restituisce `NaN`.

**LBP:** si calcola sull'intera patch (l'operatore è locale e ha bisogno del vicinato), ma l'istogramma si accumula **solo sui pixel nucleari**. `lbp_entropy` è l'entropia di Shannon in base 2 dell'istogramma normalizzato.

**Caso non definito:** patch senza alcun pixel nucleare → tutte e 6 le colonne a `NaN` (coerente con D1).

- [ ] **Step 1: scrivere i test che falliscono**

```python
# tests/test_feature_texture.py
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

TEXTURE_COLUMNS = {
    "glcm_contrast", "glcm_homogeneity", "glcm_energy",
    "lbp_entropy", "hchannel_mean", "hchannel_std",
}


@pytest.fixture(scope="module")
def features():
    spec = importlib.util.spec_from_file_location(
        "mod_features", SRC_DIR / "03_feature_extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_returns_exactly_the_six_documented_columns(features):
    h_channel = np.full((32, 32), 120, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[8:24, 8:24] = 1

    result = features.extract_texture_features(h_channel, mask)

    assert set(result) == TEXTURE_COLUMNS


def test_a_uniform_nuclear_region_has_zero_contrast_and_maximal_homogeneity(features):
    h_channel = np.full((32, 32), 120, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[8:24, 8:24] = 1

    result = features.extract_texture_features(h_channel, mask)

    assert result["glcm_contrast"] == pytest.approx(0.0)
    assert result["glcm_homogeneity"] == pytest.approx(1.0)
    assert result["hchannel_std"] == pytest.approx(0.0)
    assert result["hchannel_mean"] == pytest.approx(120.0)


def test_intensity_statistics_ignore_the_background(features):
    """Regressione della decisione D2: lo sfondo non deve entrare nelle statistiche."""
    h_channel = np.zeros((32, 32), dtype=np.uint8)
    h_channel[8:24, 8:24] = 200
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[8:24, 8:24] = 1

    result = features.extract_texture_features(h_channel, mask)

    assert result["hchannel_mean"] == pytest.approx(200.0)


def test_a_noisy_region_has_higher_contrast_than_a_uniform_one(features):
    mask = np.ones((32, 32), dtype=np.int32)
    uniform = np.full((32, 32), 120, dtype=np.uint8)
    checkerboard = np.indices((32, 32)).sum(axis=0) % 2
    noisy = np.where(checkerboard == 0, 40, 200).astype(np.uint8)

    uniform_result = features.extract_texture_features(uniform, mask)
    noisy_result = features.extract_texture_features(noisy, mask)

    assert noisy_result["glcm_contrast"] > uniform_result["glcm_contrast"]
    assert noisy_result["glcm_homogeneity"] < uniform_result["glcm_homogeneity"]


def test_every_column_is_nan_when_the_mask_has_no_nuclei(features):
    h_channel = np.full((32, 32), 120, dtype=np.uint8)
    empty_mask = np.zeros((32, 32), dtype=np.int32)

    result = features.extract_texture_features(h_channel, empty_mask)

    assert all(math.isnan(v) for v in result.values())


def test_homogeneity_and_energy_stay_within_their_theoretical_range(features):
    rng = np.random.default_rng(0)
    h_channel = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.int32)

    result = features.extract_texture_features(h_channel, mask)

    assert 0.0 < result["glcm_homogeneity"] <= 1.0
    assert 0.0 < result["glcm_energy"] <= 1.0
    assert result["glcm_contrast"] >= 0.0
    assert result["lbp_entropy"] >= 0.0
```

- [ ] **Step 2: eseguire i test e verificare che falliscano**

Run: `python -m pytest tests/test_feature_texture.py -q`
Atteso: FAIL con `NotImplementedError: STEP 4 — da implementare nella prossima iterazione`.

- [ ] **Step 3: implementare**

```python
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Parametri di tessitura — fissati qui e replicati in
# feature_extraction_metadata.json per riproducibilità (decisione D3).
GLCM_LEVELS = 64                        # quantizzazione: standard Haralick
GLCM_DISTANCES = (1,)                   # px
GLCM_ANGLES_DEG = (0, 45, 90, 135)      # mediati per invarianza rotazionale
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"


def extract_texture_features(
    h_channel_patch: np.ndarray, instance_mask: np.ndarray
) -> dict:
    """
    Descrittori di tessitura cromatinica sui soli pixel nucleari.

    GLCM, LBP e statistiche di intensità sono ristretti ai pixel appartenenti
    ai nuclei (decisione D2 del piano): calcolarli sull'intera patch
    misurerebbe anche stroma e spazio inter-nucleare, diluendo il segnale
    cromatinico che il lavoro intende quantificare.

    Args:
        h_channel_patch: H-channel CLAHE uint8 prodotto dalla Fase 1.
        instance_mask: maschera d'istanza della Fase 2 (0 = sfondo).

    Returns:
        dict con glcm_contrast, glcm_homogeneity, glcm_energy, lbp_entropy,
        hchannel_mean, hchannel_std. Tutte NaN se la maschera è vuota.
    """
    nan = float("nan")
    nuclear = instance_mask > 0

    if not nuclear.any():
        return {
            "glcm_contrast": nan, "glcm_homogeneity": nan, "glcm_energy": nan,
            "lbp_entropy": nan, "hchannel_mean": nan, "hchannel_std": nan,
        }

    h = h_channel_patch.astype(np.float64)

    # --- Statistiche di intensità sui soli pixel nucleari ---
    nuclear_values = h[nuclear]

    # --- GLCM mascherata ---
    # I livelli utili sono 1..GLCM_LEVELS-1; lo 0 è riservato allo sfondo,
    # così le coppie che lo coinvolgono sono isolabili e scartabili.
    quantized = np.floor(h / 256.0 * (GLCM_LEVELS - 1)).astype(np.uint8) + 1
    quantized[~nuclear] = 0

    glcm = graycomatrix(
        quantized,
        distances=list(GLCM_DISTANCES),
        angles=[np.deg2rad(a) for a in GLCM_ANGLES_DEG],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=False,
    )
    # Scarta le coppie che coinvolgono lo sfondo.
    # La rinormalizzazione la fa graycoprops; qui serve solo verificare che
    # resti almeno una coppia di pixel nucleari adiacenti, altrimenti
    # graycoprops restituirebbe zeri indistinguibili da una tessitura piatta.
    glcm = glcm[1:, 1:, :, :].astype(np.float64)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    if not np.all(totals > 0):
        return {
            "glcm_contrast": nan, "glcm_homogeneity": nan, "glcm_energy": nan,
            "lbp_entropy": nan,
            "hchannel_mean": round(float(nuclear_values.mean()), 4),
            "hchannel_std": round(float(nuclear_values.std()), 4),
        }
    # --- LBP: operatore su tutta la patch, istogramma sui soli nuclei ---
    lbp = local_binary_pattern(h_channel_patch, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    n_bins = LBP_POINTS + 2  # metodo 'uniform'
    histogram, _ = np.histogram(lbp[nuclear], bins=n_bins, range=(0, n_bins))
    probabilities = histogram / histogram.sum()
    non_zero = probabilities[probabilities > 0]
    lbp_entropy = float(-(non_zero * np.log2(non_zero)).sum())

    return {
        "glcm_contrast": round(float(graycoprops(glcm, "contrast").mean()), 4),
        "glcm_homogeneity": round(float(graycoprops(glcm, "homogeneity").mean()), 4),
        "glcm_energy": round(float(graycoprops(glcm, "energy").mean()), 4),
        "lbp_entropy": round(lbp_entropy, 4),
        "hchannel_mean": round(float(nuclear_values.mean()), 4),
        "hchannel_std": round(float(nuclear_values.std()), 4),
    }
```

- [ ] **Step 4: eseguire i test e verificare che passino**

Run: `python -m pytest tests/ -q`
Atteso: PASS.

- [ ] **Step 5: commit**

```bash
git add src/03_feature_extraction.py tests/test_feature_texture.py
git commit -m "feat(fase3): tessitura cromatinica GLCM/LBP ristretta ai pixel nucleari"
```

---

### Task 3 — Cablaggio in `run_fase3` e contratto delle colonne

**File:**
- Modifica: `src/03_feature_extraction.py` (nuova costante `PATCH_FEATURE_COLUMNS`)
- Modifica: `src/run_pipeline.py` (`run_fase3`)
- Test: `tests/test_patch_feature_contract.py` (nuovo)

**Interfacce prodotte:**
```python
PATCH_FEATURE_COLUMNS: tuple[str, ...]   # 47 nomi, ordine canonico del CSV
PATCH_METADATA_COLUMNS = ("image_name", "category", "target")
```

L'ordine delle colonne segue le sezioni del report: metadati, densità (3), Iwamoto (2), morfometria (32), k-NN (4), tessitura (6).

`run_fase3` deve: caricare l'H-channel da `patch.h_channel_path`, invocare `compute_knn_spatial_features` e `extract_texture_features`, unirne l'output al dizionario per patch, e scrivere il CSV usando `PATCH_METADATA_COLUMNS + PATCH_FEATURE_COLUMNS` come `fieldnames` — così un'eventuale colonna mancante o di troppo diventa un `ValueError` di `csv.DictWriter` invece di un CSV silenziosamente diverso dal contratto.

- [ ] **Step 1: scrivere il test che fallisce**

```python
# tests/test_patch_feature_contract.py
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture(scope="module")
def features():
    spec = importlib.util.spec_from_file_location(
        "mod_features", SRC_DIR / "03_feature_extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_contract_declares_exactly_47_features_and_3_metadata(features):
    assert len(features.PATCH_FEATURE_COLUMNS) == 47
    assert len(features.PATCH_METADATA_COLUMNS) == 3
    assert len(set(features.PATCH_FEATURE_COLUMNS)) == 47, "nomi duplicati nel contratto"


def test_a_full_patch_row_matches_the_declared_contract(features):
    mask = np.zeros((64, 64), dtype=np.int32)
    mask[10:20, 10:20] = 1
    mask[30:40, 30:40] = 2
    mask[10:20, 30:40] = 3
    mask[30:40, 10:20] = 4
    h_channel = np.full((64, 64), 130, dtype=np.uint8)

    nuclei = features.extract_nucleus_morphometry(mask)
    row = features.aggregate_patch_morphometry(nuclei, "patch_test", "follicular_lymphoma")
    row.update(features.compute_knn_spatial_features(nuclei))
    row.update(features.extract_texture_features(h_channel, mask))

    produced = set(row) - set(features.PATCH_METADATA_COLUMNS)
    assert produced == set(features.PATCH_FEATURE_COLUMNS)
```

- [ ] **Step 2: eseguire il test e verificare che fallisca**

Run: `python -m pytest tests/test_patch_feature_contract.py -q`
Atteso: FAIL con `AttributeError: module has no attribute 'PATCH_FEATURE_COLUMNS'`.

- [ ] **Step 3: definire il contratto e cablare `run_fase3`**

In `src/03_feature_extraction.py`, dopo `MORPHOMETRY_BASE_FEATURES`:

```python
PATCH_METADATA_COLUMNS: tuple[str, ...] = ("image_name", "category", "target")

_DENSITY_COLUMNS = ("n_nuclei", "nuclear_density_per_1000um2", "nuclear_area_fraction")
_IWAMOTO_COLUMNS = ("area_top10_mean_um2", "area_top10_short_axis_um")
_MORPHOMETRY_COLUMNS = tuple(
    f"{feat}_{stat}"
    for feat in MORPHOMETRY_BASE_FEATURES
    for stat in ("mean", "std", "skew", "cv")
)
_KNN_COLUMNS = tuple(
    f"knn{k}_dist_{stat}_um" for k in KNN_NEIGHBOR_COUNTS for stat in ("mean", "std")
)
_TEXTURE_COLUMNS = (
    "glcm_contrast", "glcm_homogeneity", "glcm_energy",
    "lbp_entropy", "hchannel_mean", "hchannel_std",
)

# Ordine canonico delle colonne del CSV per patch (47), nell'ordine delle
# sezioni di reports/fase3_report.md §2.
PATCH_FEATURE_COLUMNS: tuple[str, ...] = (
    _DENSITY_COLUMNS + _IWAMOTO_COLUMNS + _MORPHOMETRY_COLUMNS
    + _KNN_COLUMNS + _TEXTURE_COLUMNS
)
```

In `src/run_pipeline.py`, importare le nuove funzioni dal modulo 03 accanto a quelle già importate, e dentro il ciclo di `run_fase3`, dopo l'aggregazione morfometrica:

```python
                h_channel = cv2.imread(str(patch.h_channel_path), cv2.IMREAD_GRAYSCALE)
                if h_channel is None:
                    raise ValueError(f"Impossibile leggere {patch.h_channel_path.name}")

                patch_stat.update(compute_knn_spatial_features(nuclei_feat))
                patch_stat.update(extract_texture_features(h_channel, mask_16))
                patch_stat["target"] = target_from_category(category)
```

e sostituire la costruzione dei `fieldnames` del CSV per patch:

```python
        fieldnames = list(PATCH_METADATA_COLUMNS) + list(PATCH_FEATURE_COLUMNS)
```

- [ ] **Step 4: eseguire i test e verificare che passino**

Run: `python -m pytest tests/ -q`
Atteso: PASS. Il test di integrazione `test_pipeline_end_to_end.py` deve continuare a passare: verifica implicitamente che il cablaggio funzioni su immagini reali.

- [ ] **Step 5: commit**

```bash
git add src/03_feature_extraction.py src/run_pipeline.py tests/test_patch_feature_contract.py
git commit -m "feat(fase3): cablaggio k-NN e tessitura in run_fase3 con contratto a 47 colonne"
```

---

### Task 4 — `feature_extraction_metadata.json`

**File:**
- Modifica: `src/run_pipeline.py` (`run_fase3`)
- Test: `tests/test_pipeline_end_to_end.py` (estensione)

**Contenuto del file** (scritto in `data/fase3_features/`):

```json
{
  "fase": 3,
  "generato_il": "<ISO 8601>",
  "calibrazione": {"microns_per_pixel": 0.23, "patch_size_px": 224, "patch_area_um2": 2654.31},
  "conteggi": {"patch_processate": 600, "patch_in_errore": 0, "nuclei_totali": 94042},
  "feature": {"n_feature": 47, "n_metadati": 3, "colonne": ["..."]},
  "parametri_glcm": {"levels": 64, "distances": [1], "angles_deg": [0, 45, 90, 135], "mascherato_sui_nuclei": true},
  "parametri_lbp": {"points": 8, "radius": 1, "method": "uniform", "mascherato_sui_nuclei": true},
  "parametri_knn": {"k": [1, 3], "valore_se_non_definito": "NaN"},
  "decisioni": {"D1": "NaN sui k-NN non definiti", "D2": "tessitura sui soli pixel nucleari", "D3": "GLCM a 64 livelli"},
  "ambiente": {"python": "...", "numpy": "...", "scipy": "...", "scikit-image": "..."},
  "tempo_esecuzione_s": 0.0
}
```

- [ ] **Step 1: aggiungere il test che fallisce** in `tests/test_pipeline_end_to_end.py`

```python
def test_fase3_writes_a_reproducible_metadata_file(pipeline):
    import json

    module, stems = pipeline
    metadata_path = module.FASE3_DIR / "feature_extraction_metadata.json"

    assert metadata_path.exists(), "metadata di riproducibilità non generato"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["conteggi"]["patch_processate"] == sum(len(s) for s in stems.values())
    assert metadata["feature"]["n_feature"] == 47
    assert metadata["parametri_glcm"]["levels"] == 64
    assert metadata["parametri_glcm"]["mascherato_sui_nuclei"] is True
    assert metadata["calibrazione"]["microns_per_pixel"] == 0.23
```

- [ ] **Step 2: eseguire e verificare il fallimento**

Run: `python -m pytest tests/test_pipeline_end_to_end.py -q`
Atteso: FAIL con "metadata di riproducibilità non generato".

- [ ] **Step 3: implementare la scrittura** in `run_fase3`, dopo il salvataggio dei due CSV, usando `json.dump(..., indent=2, ensure_ascii=False)` e `importlib.metadata.version` per le versioni delle librerie.

- [ ] **Step 4: eseguire i test**

Run: `python -m pytest tests/ -q` → PASS.

- [ ] **Step 5: commit**

```bash
git add src/run_pipeline.py tests/test_pipeline_end_to_end.py
git commit -m "feat(fase3): metadata JSON di riproducibilità per l'estrazione delle feature"
```

---

### Task 5 — Esecuzione sull'intero dataset e QA

**File:**
- Genera: `data/fase3_features/features_nuclei_all.csv`, `features_patches_master.csv`, `feature_extraction_metadata.json`
- Genera: `img/fase3/morphometry_regions_preview.png`

- [ ] **Step 1: eseguire la Fase 3**

```bash
python src/run_pipeline.py --fase 3
```

- [ ] **Step 2: QA sull'output** — verificare e annotare:

| Controllo | Atteso |
|---|---|
| Righe in `features_patches_master.csv` | 600 (300 per classe) |
| Colonne | 50 |
| Righe in `features_nuclei_all.csv` | ~94.042, coerente con `centroids_all.csv` |
| Patch in errore | 0 |
| Colonne a varianza zero | nessuna (se presenti, indagare prima di proseguire) |
| Patch con k-NN `NaN` | contare per categoria — dato da riportare nel report §3.1 |
| `NaN` inattesi fuori dalle colonne k-NN/tessitura | nessuno |

- [ ] **Step 3: commit dei dati generati**

```bash
git add data/fase3_features img/fase3
git commit -m "data(fase3): estrazione biomarcatori su 600 patch"
```

---

### Task 6 — Analisi di separabilità statistica FL vs REACTIVE

**File:**
- Crea: `src/feature_analysis.py`
- Crea: `tests/test_feature_analysis.py`
- Genera: `data/fase3_features/separability_tests.csv`

**Interfacce prodotte:**
```python
def describe_by_class(df: pd.DataFrame) -> pd.DataFrame
def separability_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame
```

`separability_tests` restituisce una riga per feature con: `feature`, `mean_fl`, `mean_reactive`, `test` (`"welch_t"` o `"mann_whitney_u"`), `statistic`, `p_raw`, `p_fdr`, `effect_size`, `effect_size_type` (`"cohens_d"` o `"rank_biserial"`), `significant` (su `p_fdr`).

**Scelta del test:** Shapiro–Wilk su ciascun gruppo; se entrambi risultano normali (p > 0.05) si usa il t-test di Welch, altrimenti Mann–Whitney U. Le feature con `NaN` sono confrontate sui soli valori validi, riportando il numero di osservazioni usate.

**Correzione FDR** (D4): `scipy.stats.false_discovery_control(p_raw, method="bh")`.

- [ ] **Step 1: scrivere i test che falliscono** — casi: due gruppi chiaramente separati devono risultare significativi con effect size grande; due gruppi identici non significativi; `p_fdr >= p_raw` per ogni feature; una feature con `NaN` non deve far fallire l'analisi; il numero di righe in output deve essere 47.

- [ ] **Step 2: eseguire e verificare il fallimento** — `ModuleNotFoundError: No module named 'feature_analysis'`.

- [ ] **Step 3: implementare** `src/feature_analysis.py`.

- [ ] **Step 4: eseguire i test** → PASS.

- [ ] **Step 5: generare la tabella** e commit.

```bash
git add src/feature_analysis.py tests/test_feature_analysis.py data/fase3_features/separability_tests.csv
git commit -m "feat(fase3): test di separabilità FL vs REACTIVE con correzione FDR"
```

---

### Task 7 — Figure per la tesi

**File:**
- Modifica: `src/feature_analysis.py` (funzioni di plotting)
- Genera: `img/fase3/boxplot_top_features.png`, `img/fase3/correlation_heatmap.png`, `img/fase3/knn_distribution.png`

Figure previste:
1. **Boxplot/violin** delle prime 6 feature per effect size, FL vs REACTIVE affiancate.
2. **Heatmap di correlazione** (Spearman) fra le 47 feature — serve anche ad anticipare la multicollinearità in Fase 4.
3. **Distribuzione delle distanze k-NN** per classe, che è il risultato micro-spaziale caratteristico di questo lavoro.

Tutte a 300 dpi, con assi etichettati in unità fisiche.

- [ ] **Step 1: generare le figure**
- [ ] **Step 2: verificare** che ogni PNG esista e superi i 50 KB
- [ ] **Step 3: commit**

```bash
git add src/feature_analysis.py img/fase3
git commit -m "feat(fase3): figure di separabilità e correlazione per la tesi"
```

---

### Task 8 — Aggiornamento della documentazione

**File:**
- Modifica: `reports/fase3_report.md` — §2.5 (definizione operativa k-NN), §2.6 (dichiarare il mascheramento, D2), §3.1–3.3 (risultati reali), §4 (figure), §1.1 e §7 (divergenza dalla proposta approvata, D7)
- Modifica: `README.md` — struttura repository e risultati salienti della Fase 3
- Modifica: `reports/fase3_implementation_plan.md` — spuntare i task completati

Il punto §1.1/§7 su D7 è quello che verrà riportato nella tesi: deve dire esplicitamente che la proposta approvata richiedeva Delaunay, MST, k=5 e CIE-LAB, che l'esclusione è una scelta motivata dalla scala del campo visivo, e che l'analisi esplorativa di conferma è rinviata a un'estensione futura.

- [ ] **Step 1: aggiornare il report con i numeri reali**
- [ ] **Step 2: aggiornare il README**
- [ ] **Step 3: rileggere il documento verificando che nessun numero sia rimasto come segnaposto**
- [ ] **Step 4: commit**

```bash
git add reports README.md
git commit -m "docs(fase3): risultati, figure e motivazione delle esclusioni metodologiche"
```

---

## 5. Criteri di completamento della Fase 3

La Fase 3 è chiusa quando tutte queste condizioni sono vere contemporaneamente:

- [ ] `features_patches_master.csv` esiste con 600 righe × 50 colonne
- [ ] `features_nuclei_all.csv` è coerente con `centroids_all.csv` sul numero di nuclei
- [ ] `feature_extraction_metadata.json` documenta tutti i parametri di D1, D2, D3
- [ ] `separability_tests.csv` riporta p-value grezzi e corretti FDR per tutte le 47 feature
- [ ] `python -m pytest tests/ -q` interamente verde
- [ ] `reports/fase3_report.md` non contiene più segnaposto
- [ ] La divergenza D7 dalla proposta approvata è documentata in forma citabile nella tesi
