"""
===============================================================================
block_structure.py — L'ordine di numerazione delle patch conserva struttura?
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
===============================================================================
LA DOMANDA. La Fase 4 valuta ogni modello due volte, con split casuale e con
split a blocchi contigui di indice (spec D1). La validazione a blocchi poggia su
una premessa: che patch con indici vicini tendano a provenire dallo stesso caso,
perche' l'esportazione del dataset e' avvenuta caso per caso. Le etichette di
caso non esistono nel dataset Zenodo, quindi la premessa non e' verificabile in
modo diretto. Puo' pero' essere messa alla prova in modo indiretto.

IL METODO. Se l'ordine di numerazione conservasse struttura, patch adiacenti
dovrebbero somigliarsi piu' di due patch prese a caso. Si misura quindi, dentro
ciascuna classe:

    rapporto = distanza media fra coppie adiacenti
               ------------------------------------
               distanza media fra coppie qualsiasi

Le distanze sono euclidee nello spazio dei biomarcatori standardizzati. La
standardizzazione avviene DENTRO la classe, non sul dataset intero: altrimenti
la separazione fra le due classi dominerebbe le distanze, e si misurerebbe la
biologia invece dell'ordine dei file.

Un rapporto pari a 1 significa che l'ordine non porta informazione. Un rapporto
minore di 1 significa che le patch vicine si somigliano.

IL TEST. Il rapporto da solo non basta: serve sapere quanto sarebbe raro
ottenerlo per caso. Si permuta ripetutamente l'ordine delle patch e si ricalcola
il rapporto, costruendo la distribuzione nulla. Il valore p e' la frazione di
permutazioni che arriva a un rapporto basso quanto quello osservato, calcolata
come (1 + successi) / (1 + permutazioni): l'aggiunta di uno evita di riportare
p = 0, che una stima per campionamento non puo' sostenere (Phipson & Smyth,
2010, Stat. Appl. Genet. Mol. Biol. 9(1), DOI: 10.2202/1544-6115.1585).

COME SI LEGGE. Un rapporto nettamente sotto 1 con p piccolo dice che l'ordine di
numerazione conserva struttura, e quindi che raggruppare per prossimita' di
indice non e' arbitrario. NON dice che i blocchi coincidano con i casi clinici:
quella resta un'ipotesi non verificabile senza le etichette, e come tale va
dichiarata nella tesi. I blocchi restano una sonda, non un'identificazione di
paziente.

PERCHE' ESISTE QUESTO MODULO. I valori del rapporto erano gia' citati nella spec
di Fase 4 e nell'intestazione di 04_classification.py, ma nessuno script nel
repository li produceva: erano un'affermazione, non un risultato riproducibile.
Sono il pilastro della validazione a blocchi, quindi devono essere ricalcolabili
da chiunque.

Esecuzione:
    python src/block_structure.py
===============================================================================
"""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

BASE_DIR = _SRC_DIR.parent
FEATURES_CSV = BASE_DIR / "data" / "fase3_features" / "features_patches_master.csv"
FASE4_DIR = BASE_DIR / "data" / "fase4_classification"

SEED = 42
N_PERMUTATIONS = 2000

# Stessa forma di nome richiesta da contiguous_blocks(): il prefisso identifica
# la classe, il numero fra parentesi l'ordine di esportazione.
_INDEX_PATTERN = re.compile(r"^(?P<prefix>.*?)\s*\((?P<index>\d+)\)\s*$")


def _classification():
    """
    Il modulo della Fase 4, che possiede il caricamento della matrice.

    Il nome comincia con una cifra, quindi non e' importabile con `import`:
    stessa soluzione adottata in stain_robustness.py.
    """
    spec = importlib.util.spec_from_file_location(
        "classification_module", _SRC_DIR / "04_classification.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_index(image_name: str) -> tuple[str, int]:
    """
    Ricava prefisso e indice numerico da un nome di patch.

    Raises:
        ValueError: se il nome non contiene un indice. L'analisi dipende
            dall'ordine di numerazione: un nome non conforme e' un errore
            visibile, non una patch da collocare a caso.
    """
    match = _INDEX_PATTERN.match(str(image_name))
    if match is None:
        raise ValueError(
            f"impossibile ricavare l'indice numerico da '{image_name}': l'analisi "
            "della struttura richiede nomi nella forma '<prefisso> (<numero>)'."
        )
    return match["prefix"], int(match["index"])


def _standardise(X: np.ndarray) -> np.ndarray:
    """
    Standardizza le colonne, scartando quelle costanti.

    Una colonna a varianza nulla non distingue nessuna coppia di patch e
    dividerla per la propria deviazione standard produrrebbe NaN.
    """
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    return (X - X.mean(axis=0)) / X.std(axis=0)


def _adjacency_ratio(Z: np.ndarray, mean_all_pairs: float) -> float:
    """Distanza media fra righe consecutive, in rapporto a quella fra tutte."""
    return float(np.linalg.norm(Z[1:] - Z[:-1], axis=1).mean() / mean_all_pairs)


def index_order_similarity(
    X: np.ndarray,
    image_names,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Misura quanta struttura conserva l'ordine di numerazione, classe per classe.

    Args:
        X: matrice (n_patch, n_feature) dei biomarcatori.
        image_names: nomi delle patch, nella forma "<prefisso> (<indice>)".
        n_permutations: estrazioni della distribuzione nulla.
        seed: seme del generatore, per riproducibilita'.

    Returns:
        Una riga per prefisso, con distanze medie, rapporto e valore p.

    Raises:
        ValueError: se un nome non contiene un indice numerico, o se una classe
            ha meno di tre patch (con due il rapporto vale uno per costruzione).
    """
    X = np.asarray(X, dtype=float)
    parsed = [parse_index(name) for name in image_names]
    rng = np.random.default_rng(seed)

    rows = []
    for prefix in sorted({p for p, _ in parsed}):
        positions = [i for i, (p, _) in enumerate(parsed) if p == prefix]
        positions.sort(key=lambda i: parsed[i][1])
        if len(positions) < 3:
            raise ValueError(
                f"la classe '{prefix}' ha {len(positions)} patch: ne servono almeno 3 "
                "perche' il rapporto fra coppie adiacenti e coppie qualsiasi abbia senso."
            )

        Z = _standardise(X[positions])

        upper = np.triu_indices(len(Z), k=1)
        all_pairs = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)[upper]
        mean_all_pairs = float(all_pairs.mean())
        mean_adjacent = float(np.linalg.norm(Z[1:] - Z[:-1], axis=1).mean())
        ratio = mean_adjacent / mean_all_pairs

        # Distribuzione nulla: lo stesso rapporto quando l'ordine non conta.
        # mean_all_pairs non dipende dall'ordine, quindi si ricalcola solo il
        # numeratore.
        at_least_as_extreme = sum(
            _adjacency_ratio(Z[rng.permutation(len(Z))], mean_all_pairs) <= ratio
            for _ in range(n_permutations)
        )
        p_value = (1 + at_least_as_extreme) / (1 + n_permutations)

        rows.append({
            "prefix": prefix,
            "n_patches": len(Z),
            "n_features": Z.shape[1],
            "mean_distance_adjacent": round(mean_adjacent, 4),
            "mean_distance_all_pairs": round(mean_all_pairs, 4),
            "ratio": round(ratio, 4),
            "p_permutation": round(p_value, 6),
            "n_permutations": n_permutations,
        })

    return pd.DataFrame(rows)


def main() -> None:
    data = _classification().load_feature_matrix(FEATURES_CSV)

    print("[Struttura] Rapporto fra distanza media di coppie adiacenti e coppie qualsiasi.")
    print(f"[Struttura] {N_PERMUTATIONS} permutazioni per la distribuzione nulla, seed {SEED}.")

    table = index_order_similarity(data.X, data.image_names)

    FASE4_DIR.mkdir(parents=True, exist_ok=True)
    destination = FASE4_DIR / "block_structure_evidence.csv"
    table.to_csv(destination, index=False)

    print()
    print(table.to_string(index=False))
    print()
    print(f"[OK] scritto {destination.relative_to(BASE_DIR)}")
    print("[Nota] Un rapporto sotto 1 dice che l'ordine di numerazione conserva")
    print("       struttura. Non dice che i blocchi coincidano con i casi clinici.")


if __name__ == "__main__":
    main()
