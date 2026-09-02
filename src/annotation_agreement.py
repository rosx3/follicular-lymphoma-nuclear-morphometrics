"""
===============================================================================
annotation_agreement.py — Il riferimento Cellpose regge a un conteggio umano?
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
===============================================================================
LA DOMANDA. Tutta la validazione della Fase 2 misura il Watershed contro
Cellpose. Ma Cellpose non e' mai stato verificato su queste immagini, e la
discrepanza aperta e' di detection: Cellpose rileva 196.9 nuclei per patch, il
Watershed 167.5. Il report chiama quel divario "sotto-rilevazione del
Watershed", ma e' sotto-rilevazione RISPETTO A CELLPOSE, e nessuno sa quale dei
due sia piu' vicino al vero.

IL METODO. Un lettore umano marca con un punto il centro di ogni nucleo, alla
cieca, su tutte e 10 le patch da cui provengono i numeri pubblicati. Ogni punto
viene poi confrontato con le maschere dei due algoritmi.

Il criterio di accoppiamento e' il CONTENIMENTO: un nucleo umano e' "trovato" se
il suo punto cade dentro un'istanza segmentata. E' preferibile alla distanza dal
centroide perche' verifica la cosa giusta, cioe' se l'algoritmo abbia segmentato
un oggetto proprio li', senza introdurre una soglia arbitraria in pixel.

Da qui si leggono tre cose distinte, che l'accordo aggregato confonderebbe:
  - nuclei umani NON coperti da alcuna istanza: l'algoritmo li ha persi;
  - istanze che contengono PIU' di un punto umano: l'algoritmo ha fuso nuclei
    che il lettore distingueva;
  - istanze che non contengono ALCUN punto umano: l'algoritmo ha segmentato
    qualcosa che il lettore non ha riconosciuto come nucleo, oppure il lettore
    l'ha mancato.

COSA QUESTO CONFRONTO NON E'. Il lettore non e' un patologo: non esiste un
patologo in questo progetto. Il riferimento umano prodotto qui e' quindi NON
ESPERTO, e non e' una ground truth clinica. Serve a rilevare modi di fallire
grossolani, non a stabilire il conteggio nucleare corretto in senso diagnostico.
La regola applicata sui nuclei addossati (due clic quando il contorno e'
distinguibile) va riportata insieme ai risultati: una regola piu' permissiva
alzerebbe i conteggi umani, una piu' restrittiva li abbasserebbe.

Esecuzione:
    python src/annotation_agreement.py
===============================================================================
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

BASE_DIR = _SRC_DIR.parent
FASE1_DIR = BASE_DIR / "data" / "fase1_preprocessing"
CELLPOSE_DIR = BASE_DIR / "data" / "ground_truth" / "cellpose_v4"
ANNOTATION_DIR = BASE_DIR / "data" / "annotazione_manuale"
IMG_DIR = BASE_DIR / "img" / "fase2" / "annotazione"

CATEGORY_DIR = {
    "Follicular Lymphoma": "follicular_lymphoma",
    "Reactive Tissue": "reactive_tissue",
}

# Gli stessi parametri con cui sono state prodotte le 600 maschere del dataset.
# Non sono negoziabili: cambiarli qui misurerebbe un Watershed diverso da quello
# su cui poggia la tesi.
WS_PARAMS = dict(
    min_distance=7, min_area_px=15, max_area_px=2500,
    marker_method="relative_threshold", peak_threshold_rel=0.15,
)

# Due punti piu' vicini di questo (in px di schermo) sono lo stesso clic: serve a
# separare i dubbi dai certi, dato che il file dei dubbi e' un sovrainsieme.
SAME_CLICK_PX = 8.0


def _segmentation():
    """Il modulo della Fase 2. Il nome comincia con una cifra, serve importlib."""
    spec = importlib.util.spec_from_file_location(
        "segmentation_module", _SRC_DIR / "02_segmentation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def screen_to_patch(points: np.ndarray, upscale: int) -> np.ndarray:
    """
    Converte le coordinate cliccate sull'immagine ingrandita in coordinate della
    patch originale.

    Si usa la convenzione di `cv2.resize`, che allinea i CENTRI dei pixel:

        src = (dst + 0.5) / fattore - 0.5

    La divisione ingenua per il fattore sbaglia di mezzo pixel sorgente. E' poco
    rispetto a un nucleo di 13-26 px, ma e' gratis farla giusta.
    """
    return (points + 0.5) / upscale - 0.5


def _read_points(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    missing = {"X", "Y"} - set(table.columns)
    if missing:
        raise ValueError(f"{path.name}: colonne mancanti {sorted(missing)}")
    return table[["X", "Y"]].to_numpy(dtype=float)


def split_certain_and_doubtful(
    points_file: Path, doubts_file: Path | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa i punti certi dai dubbi.

    Il file dei dubbi e' prodotto senza deselezionare i punti gia' messi, quindi
    contiene i certi PIU' i dubbi. I dubbi si ricavano per differenza
    geometrica: sono i punti del secondo file che non compaiono nel primo.
    """
    certain = _read_points(points_file)
    if doubts_file is None or not doubts_file.exists():
        return certain, np.empty((0, 2))

    combined = _read_points(doubts_file)
    if len(combined) < len(certain):
        raise ValueError(
            f"{doubts_file.name} ha meno punti di {points_file.name}: i due file "
            "sembrano invertiti. Il file dei dubbi deve contenere i certi piu' i dubbi."
        )

    distance = np.linalg.norm(combined[:, None, :] - certain[None, :, :], axis=2)
    return certain, combined[distance.min(axis=1) >= SAME_CLICK_PX]


def compare_points_to_mask(points_patch: np.ndarray, mask: np.ndarray) -> dict:
    """
    Confronta i punti umani con una maschera d'istanza.

    Args:
        points_patch: (N, 2) coordinate (x, y) nello spazio della patch.
        mask: maschera d'istanza int (0 = sfondo).

    Returns:
        dict con i conteggi dei tre modi di divergere.
    """
    height, width = mask.shape
    columns = np.clip(np.floor(points_patch[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.floor(points_patch[:, 1]).astype(int), 0, height - 1)

    labels_at_points = mask[rows, columns]
    covered = labels_at_points > 0

    hit_labels = labels_at_points[covered]
    unique_hit, counts = np.unique(hit_labels, return_counts=True)

    n_instances = int(len(np.unique(mask)) - (1 if (mask == 0).any() else 0))
    merged_instances = int((counts > 1).sum())
    points_in_merged = int(counts[counts > 1].sum())

    return {
        "n_punti_umani": int(len(points_patch)),
        "n_istanze": n_instances,
        "umani_trovati": int(covered.sum()),
        "umani_persi": int((~covered).sum()),
        "istanze_senza_punto": int(n_instances - len(unique_hit)),
        "istanze_con_piu_punti": merged_instances,
        "umani_in_istanze_fuse": points_in_merged,
        "recall_umano": round(float(covered.mean()), 4) if len(points_patch) else float("nan"),
    }


def _watershed_mask(image_name: str, category: str) -> np.ndarray:
    h_path = (
        FASE1_DIR / CATEGORY_DIR[category] / "h_channel" / f"{image_name}_hchannel.png"
    )
    h_channel = cv2.imread(str(h_path), cv2.IMREAD_GRAYSCALE)
    if h_channel is None:
        raise FileNotFoundError(f"canale H mancante: {h_path}")
    mask, _ = _segmentation().segment_nuclei_watershed(h_channel, **WS_PARAMS)
    return mask


def _cellpose_mask(image_name: str) -> np.ndarray:
    path = CELLPOSE_DIR / f"{image_name}_cellpose_gt.png"
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"maschera Cellpose mancante: {path}")
    return mask.astype(np.int32)


def plot_overlay(
    image_name: str,
    category: str,
    points_patch: np.ndarray,
    masks: dict[str, np.ndarray],
    destination: Path,
) -> None:
    """
    Sovrappone i punti umani ai contorni dei due algoritmi, un pannello ciascuno.

    Serve a due cose diverse. La prima e' la figura della tesi. La seconda, piu'
    utile adesso, e' il controllo visivo: i disaccordi contati dalla tabella qui
    si vedono, e chi ha annotato puo' verificare se siano errori suoi o
    dell'algoritmo.

    I nuclei che l'algoritmo ha FUSO, cioe' le istanze che contengono piu' di un
    punto umano, sono evidenziati: sono il disaccordo piu' informativo, perche'
    riguardano proprio la separazione dei nuclei addossati.
    """
    import matplotlib.pyplot as plt

    rgb_path = FASE1_DIR / CATEGORY_DIR[category] / "rgb_normalized" / f"{image_name}_norm.png"
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)

    figure, axes = plt.subplots(1, len(masks), figsize=(7.5 * len(masks), 8))
    axes = np.atleast_1d(axes)

    for axis, (algorithm, mask) in zip(axes, masks.items()):
        axis.imshow(rgb)

        columns = np.clip(np.floor(points_patch[:, 0]).astype(int), 0, mask.shape[1] - 1)
        rows = np.clip(np.floor(points_patch[:, 1]).astype(int), 0, mask.shape[0] - 1)
        labels_at_points = mask[rows, columns]
        hit, counts = np.unique(labels_at_points[labels_at_points > 0], return_counts=True)
        merged = set(hit[counts > 1].tolist())

        for instance_id in np.unique(mask):
            if instance_id == 0:
                continue
            contours, _ = cv2.findContours(
                np.uint8(mask == instance_id), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            fused = instance_id in merged
            for contour in contours:
                closed = np.vstack([contour[:, 0, :], contour[0, 0, :]])
                axis.plot(
                    closed[:, 0], closed[:, 1],
                    color="#e8590c" if fused else "#2f9e44",
                    linewidth=1.6 if fused else 0.7,
                )

        covered = labels_at_points > 0
        axis.scatter(
            points_patch[covered, 0], points_patch[covered, 1],
            s=14, c="white", edgecolors="black", linewidths=0.4, zorder=3,
        )
        axis.scatter(
            points_patch[~covered, 0], points_patch[~covered, 1],
            s=42, marker="x", c="#e03131", linewidths=1.6, zorder=4,
        )

        axis.set_title(
            f"{algorithm} — {len(np.unique(mask)) - 1} istanze, "
            f"{int((~covered).sum())} nuclei umani persi, {len(merged)} fusioni",
            fontsize=11,
        )
        axis.set_xticks([]); axis.set_yticks([])

    figure.suptitle(
        f"{image_name} — punti umani su contorni algoritmici\n"
        "bianco: nucleo umano riconosciuto · rosso: nucleo umano non coperto · "
        "arancio: istanza che fonde piu' nuclei umani",
        fontsize=12,
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """
    Aggrega il confronto sulle immagini annotate.

    Si riportano SOLO le grandezze indipendenti dalla soglia di visibilita'
    dell'annotatore, cioe' quelle calcolate sui punti marcati: recall e fusioni.
    Il conteggio delle istanze senza punto umano e' deliberatamente escluso dalla
    sintesi: mescola nuclei pallidi esclusi di proposito con eventuali falsi
    positivi, e non li distingue (vedi il protocollo in
    data/annotazione_manuale/README.md).

    Il confronto appaiato e' lecito perche' i due algoritmi sono valutati sulle
    stesse immagini e sugli stessi punti. Con 10 immagini il p minimo di un
    Wilcoxon a due code e' 2/2^10 = 0.002.
    """
    from scipy.stats import wilcoxon

    rows = []
    for algorithm, group in table.groupby("algoritmo", sort=False):
        recall = group["recall_umano"]
        interval = 1.96 * recall.std(ddof=1) / np.sqrt(len(recall))
        rows.append({
            "algoritmo": algorithm,
            "n_immagini": len(group),
            "nuclei_umani": int(group["n_punti_umani"].sum()),
            "nuclei_coperti": int(group["umani_trovati"].sum()),
            "copertura": round(float(group["umani_trovati"].sum() / group["n_punti_umani"].sum()), 4),
            "recall_medio": round(float(recall.mean()), 4),
            "recall_ic95_inf": round(float(recall.mean() - interval), 4),
            "recall_ic95_sup": round(float(recall.mean() + interval), 4),
            "fusioni": int(group["istanze_con_piu_punti"].sum()),
            "immagini_con_fusioni": int((group["istanze_con_piu_punti"] > 0).sum()),
            "nuclei_in_fusioni": int(group["umani_in_istanze_fuse"].sum()),
            "quota_nuclei_in_fusioni": round(
                float(group["umani_in_istanze_fuse"].sum() / group["n_punti_umani"].sum()), 4
            ),
        })

    summary = pd.DataFrame(rows)

    pivot = table.pivot(index="image_name", columns="algoritmo")
    if {"watershed", "cellpose"} <= set(table["algoritmo"]):
        for metric, better_is_low in (("recall_umano", False), ("istanze_con_piu_punti", True)):
            ws = pivot[(metric, "watershed")]
            cp = pivot[(metric, "cellpose")]
            wins = int((ws < cp).sum()) if not better_is_low else int((ws > cp).sum())
            summary.loc[summary.algoritmo == "watershed", f"{metric}_peggio_in"] = wins
            summary.loc[summary.algoritmo == "watershed", f"{metric}_p"] = round(
                float(wilcoxon(ws, cp).pvalue), 4
            )

    return summary


def _annotation_files() -> list[tuple[dict, Path, Path | None]]:
    """Accoppia ogni CSV di annotazione all'immagine da cui proviene."""
    metadata = json.loads((ANNOTATION_DIR / "immagini_metadata.json").read_text(encoding="utf-8"))
    by_stem = {Path(item["file"]).stem.upper(): item for item in metadata["immagini"]}

    found = []
    for csv_path in sorted(ANNOTATION_DIR.glob("*.csv")):
        name = csv_path.stem.upper()
        if not name.endswith("_PUNTI"):
            continue
        stem = name[: -len("_PUNTI")]
        if stem not in by_stem:
            raise ValueError(
                f"{csv_path.name} non corrisponde a nessuna immagine di "
                f"immagini_metadata.json (atteso uno fra {sorted(by_stem)})"
            )
        # Il file dei dubbi va cercato per prefisso, non ricavato per
        # sostituzione: i nomi arrivano da Fiji, che a volte maiuscolizza, e
        # dalla mano di chi salva, che puo' scrivere "dubbi" o "dubbii". Una
        # sostituzione letterale fallirebbe in silenzio, riportando zero dubbi
        # dove ce ne sono.
        doubts = next(
            (
                candidate
                for candidate in sorted(ANNOTATION_DIR.glob("*.csv"))
                if candidate.stem.upper().startswith(f"{stem}_DUBBI")
            ),
            None,
        )
        found.append((by_stem[stem], csv_path, doubts))

    return found


def main() -> None:
    metadata = json.loads((ANNOTATION_DIR / "immagini_metadata.json").read_text(encoding="utf-8"))
    upscale = metadata["upscale"]

    annotated = _annotation_files()
    if not annotated:
        print(f"[INFO] nessun file *_punti.csv in {ANNOTATION_DIR.relative_to(BASE_DIR)}")
        return

    print(f"[Accordo] {len(annotated)} immagini annotate su {metadata['n_immagini']} disponibili.")
    print(f"[Accordo] Ingrandimento {upscale}x, accoppiamento per contenimento.")
    print()

    rows = []
    for item, points_file, doubts_file in annotated:
        certain, doubtful = split_certain_and_doubtful(points_file, doubts_file)
        points_patch = screen_to_patch(certain, upscale)

        masks = {
            "watershed": _watershed_mask(item["image_name"], item["category"]),
            "cellpose": _cellpose_mask(item["image_name"]),
        }
        for algorithm, mask in masks.items():
            rows.append({
                "image_name": item["image_name"],
                "category": item["category"],
                "algoritmo": algorithm,
                "n_dubbi_umani": int(len(doubtful)),
                **compare_points_to_mask(points_patch, mask),
            })

        plot_overlay(
            item["image_name"], item["category"], points_patch, masks,
            IMG_DIR / f"{Path(item['file']).stem}_accordo.png",
        )

    table = pd.DataFrame(rows)
    destination = ANNOTATION_DIR / "accordo_umano.csv"
    table.to_csv(destination, index=False)

    summary = summarise(table)
    summary.to_csv(ANNOTATION_DIR / "accordo_umano_sintesi.csv", index=False)

    columns = [
        "image_name", "algoritmo", "n_punti_umani", "n_istanze",
        "umani_trovati", "umani_persi", "istanze_senza_punto",
        "istanze_con_piu_punti", "recall_umano",
    ]
    print(table[columns].to_string(index=False))
    print()
    print(f"[OK] scritto {destination.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
