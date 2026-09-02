"""
Prepara le immagini per l'annotazione manuale di riferimento.

PERCHE' ESISTE. Tutta la validazione della Fase 2 poggia su Cellpose come
riferimento indipendente, e nessuno ha mai verificato che Cellpose faccia un buon
lavoro su QUESTE immagini. La discrepanza aperta e' di detection: Cellpose trova
196.9 nuclei per patch, il Watershed 167.5, e non si sa quale dei due sia piu'
vicino al vero. Un'annotazione umana e' l'unico modo per dirlo.

COSA PRODUCE. Le 10 patch che hanno gia' la maschera Cellpose salvata su disco,
intere e ingrandite, pronte per il conteggio a clic.

PERCHE' LE PATCH INTERE E NON RITAGLI. Una prima versione di questo script
esportava ritagli centrali da 112x112, per ridurre il numero di clic. La scelta
era sbagliata: gli algoritmi hanno segmentato la patch INTERA, quindi un nucleo
sul bordo del ritaglio loro lo hanno visto per intero e l'annotatore no. In un
ritaglio da 112 px con nuclei da ~20 px la fascia di bordo tocca circa un terzo
dell'area, e il confronto ne uscirebbe sporco. Il risparmio di tempo non valeva
la perdita di pulizia.

PERCHE' TUTTE E 10. Sono esattamente le patch da cui provengono i numeri
pubblicati del benchmark (Dice 0.795, AJI 0.5411, F1 0.7108). Annotandole tutte
il riferimento umano copre l'intero insieme di valutazione, e non resta nessuna
selezione da giustificare.

INGRANDIMENTO. Le patch sono esportate a 4x (896x896) con interpolazione cubica,
per rendere comodo il clic: un nucleo linfocitario di 13-26 px diventa di 52-104
px. L'interpolazione serve solo alla vista e non aggiunge informazione. La
conversione verso le coordinate originali e' esatta:

    x_patch = x_click / 4
    y_patch = y_click / 4

USO:
    python src/prepara_annotazione.py
"""

import json
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
FASE1_DIR = BASE_DIR / "data" / "fase1_preprocessing"
GT_DIR = BASE_DIR / "data" / "ground_truth"
OUTPUT_DIR = BASE_DIR / "data" / "annotazione_manuale"

CATEGORY_DIR = {
    "Follicular Lymphoma": "follicular_lymphoma",
    "Reactive Tissue": "reactive_tissue",
}

UPSCALE = 4
PATCH_SIZE = 224


def patches_with_cellpose_gt() -> set[str]:
    """Gli stem delle patch che hanno gia' una maschera Cellpose su disco."""
    return {p.name.replace("_cellpose_gt.png", "") for p in (GT_DIR / "cellpose_v4").glob("*.png")}


def select_patches() -> list[dict]:
    """
    Tutte le patch con GT Cellpose disponibile: nessuna selezione, nessun criterio
    da difendere. L'ordine e' per classe e poi per nome, non per densita': ordinare
    per densita' suggerirebbe all'annotatore quanto aspettarsi.
    """
    summary = json.loads((GT_DIR / "gt_metadata.json").read_text(encoding="utf-8"))["patches_summary"]
    available = patches_with_cellpose_gt()

    selected = [p for p in summary if p["image_name"] in available]
    if len(selected) != len(available):
        mancanti = available - {p["image_name"] for p in selected}
        raise ValueError(f"patch con GT ma assenti da gt_metadata.json: {sorted(mancanti)}")

    return sorted(selected, key=lambda p: (p["category"], p["image_name"]))


def export(selected: list[dict]) -> list[dict]:
    """Ingrandisce e salva ogni patch intera. Restituisce i metadati."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ripulisce gli esporti della versione a ritagli, per non lasciare in giro
    # file che invitano ad annotare la cosa sbagliata.
    for stale in OUTPUT_DIR.glob("*_crop.png"):
        stale.unlink()

    records = []
    for item in selected:
        category_dir = CATEGORY_DIR[item["category"]]
        source = FASE1_DIR / category_dir / "rgb_normalized" / f"{item['image_name']}_norm.png"
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(f"impossibile leggere {source}")
        if image.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"{source.name} non e' {PATCH_SIZE}x{PATCH_SIZE}: {image.shape[:2]}")

        enlarged = cv2.resize(
            image, (PATCH_SIZE * UPSCALE, PATCH_SIZE * UPSCALE), interpolation=cv2.INTER_CUBIC
        )

        # Nome senza spazi ne' parentesi: alcuni strumenti di annotazione li
        # gestiscono male, e il nome finisce dentro i file di export.
        stem = item["image_name"].replace(" ", "_").replace("(", "").replace(")", "")
        destination = OUTPUT_DIR / f"{stem}.png"
        cv2.imwrite(str(destination), enlarged)

        records.append({
            "image_name": item["image_name"],
            "category": item["category"],
            "file": destination.name,
            "patch_size_px": PATCH_SIZE,
            "upscale": UPSCALE,
            "conversione": "x_patch = x_click / upscale",
        })

    return records


def main() -> None:
    records = export(select_patches())

    metadata = {
        "scopo": "verifica umana del riferimento Cellpose, limitata alla detection",
        "n_immagini": len(records),
        "patch_size_px": PATCH_SIZE,
        "upscale": UPSCALE,
        "immagine_annotata": "RGB normalizzata (Macenko + denoising), non canale H",
        "selezione": "tutte le patch con GT Cellpose salvata: nessuna selezione",
        "avvertenza": (
            "i conteggi di riferimento non compaiono qui di proposito: leggerli "
            "prima di annotare orienterebbe l'annotatore"
        ),
        "immagini": records,
    }
    (OUTPUT_DIR / "crops_metadata.json").unlink(missing_ok=True)
    (OUTPUT_DIR / "immagini_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[OK] {len(records)} patch intere in {OUTPUT_DIR.relative_to(BASE_DIR)}")
    print(f"     {PATCH_SIZE * UPSCALE}x{PATCH_SIZE * UPSCALE} px sullo schermo, {UPSCALE}x l'originale")
    print()
    for r in records:
        short = "FL" if r["category"] == "Follicular Lymphoma" else "REACTIVE"
        print(f"  {r['file']:34s} {short}")


if __name__ == "__main__":
    main()
