"""
===============================================================================
naming.py — Convenzioni di Naming, Categorie e Risoluzione dei Percorsi
Tesi: Classificazione Linfoma Follicolare vs Tessuto Linfoide Reattivo
===============================================================================
Unica fonte di verita' per:

  1. I suffissi dei file prodotti dalle varie fasi della pipeline
     (<stem>_norm.png, <stem>_hchannel.png, <stem>_mask.png, <stem>_overlay.png).
  2. Le etichette canoniche di categoria e la loro conversione in target binario.

MOTIVAZIONE (audit Fase 3, problemi B1 e B2):
  - B1: le fasi scrivevano e rileggevano i file ricostruendone i nomi con regole
    diverse, sparse in piu' punti di run_pipeline.py. In particolare la Fase 3
    cercava "<stem>.png" nella cartella rgb_normalized/, dove i file reali sono
    "<stem>_norm.png": il percorso non esisteva e l'anteprima veniva saltata in
    silenzio. Centralizzando qui le regole, un solo punto definisce i nomi e i
    test possono verificarli contro i file realmente presenti su disco.
  - B2: la categoria era rappresentata con tre convenzioni diverse
    ("FL", "Follicular Lymphoma", "follicular_lymphoma") in punti diversi della
    pipeline. La convenzione canonica adottata e' quella dichiarata in
    reports/fase3_report.md Sezione 2.1 e coincide con i nomi delle directory
    del dataset.
===============================================================================
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Suffissi dei file per fase
# ---------------------------------------------------------------------------
RGB_NORM_SUFFIX = "_norm.png"       # Fase 1 — RGB normalizzata Macenko + bilaterale
H_CHANNEL_SUFFIX = "_hchannel.png"  # Fase 1 — canale Ematossilina + CLAHE
MASK_SUFFIX = "_mask.png"           # Fase 2 — maschera d'istanza 16-bit
OVERLAY_SUFFIX = "_overlay.png"     # Fase 2 — overlay visivo dei contorni

# ---------------------------------------------------------------------------
# Categorie canoniche
# ---------------------------------------------------------------------------
CATEGORY_FL = "follicular_lymphoma"
CATEGORY_REACTIVE = "reactive_tissue"

CATEGORIES: tuple[str, ...] = (CATEGORY_FL, CATEGORY_REACTIVE)

# Etichetta breve usata solo per i messaggi a schermo e le legende dei grafici.
CATEGORY_SHORT_LABELS: dict[str, str] = {
    CATEGORY_FL: "FL",
    CATEGORY_REACTIVE: "REACTIVE",
}

# target binario per la Fase 4 (1 = patologico, 0 = reattivo)
CATEGORY_TARGETS: dict[str, int] = {
    CATEGORY_FL: 1,
    CATEGORY_REACTIVE: 0,
}

# Alias storici presenti nei file gia' prodotti (es. centroids_all.csv della
# Fase 2 usa "Follicular Lymphoma" / "Reactive Tissue"). La chiave e' la forma
# ridotta ai soli caratteri alfanumerici minuscoli.
_CATEGORY_ALIASES: dict[str, str] = {
    "fl": CATEGORY_FL,
    "follicularlymphoma": CATEGORY_FL,
    "reactive": CATEGORY_REACTIVE,
    "reactivetissue": CATEGORY_REACTIVE,
}


# ---------------------------------------------------------------------------
# Costruzione dei nomi di file a partire dallo stem dell'immagine originale
# ---------------------------------------------------------------------------
def rgb_normalized_name(stem: str) -> str:
    """Nome del file RGB normalizzato (Fase 1) per l'immagine `stem`."""
    return f"{stem}{RGB_NORM_SUFFIX}"


def h_channel_name(stem: str) -> str:
    """Nome del file H-channel CLAHE (Fase 1) per l'immagine `stem`."""
    return f"{stem}{H_CHANNEL_SUFFIX}"


def mask_name(stem: str) -> str:
    """Nome della maschera d'istanza 16-bit (Fase 2) per l'immagine `stem`."""
    return f"{stem}{MASK_SUFFIX}"


def overlay_name(stem: str) -> str:
    """Nome dell'overlay visivo (Fase 2) per l'immagine `stem`."""
    return f"{stem}{OVERLAY_SUFFIX}"


def _stem_from_name(path_or_name: str, suffix: str) -> str:
    name = Path(path_or_name).name
    if not name.endswith(suffix):
        raise ValueError(f"Nome file non conforme alla convenzione '{suffix}': {name}")
    return name[: -len(suffix)]


def stem_from_mask_name(mask_path_or_name: str) -> str:
    """
    Ricava lo stem dell'immagine originale dal nome (o percorso) di una maschera.

    Esempio: "data/.../FL_examples (1)_mask.png" -> "FL_examples (1)"
    """
    return _stem_from_name(mask_path_or_name, MASK_SUFFIX)


def stem_from_h_channel_name(h_path_or_name: str) -> str:
    """
    Ricava lo stem dell'immagine originale dal nome (o percorso) di un H-channel.

    Usato dalla Fase 2, che itera gli H-channel prodotti dalla Fase 1.
    Esempio: "data/.../FL_examples (1)_hchannel.png" -> "FL_examples (1)"
    """
    return _stem_from_name(h_path_or_name, H_CHANNEL_SUFFIX)


# ---------------------------------------------------------------------------
# Normalizzazione delle categorie
# ---------------------------------------------------------------------------
def normalize_category(raw: str) -> str:
    """
    Converte una qualunque delle varianti storiche di etichetta di categoria
    nella forma canonica ("follicular_lymphoma" / "reactive_tissue").

    Raises:
        ValueError: se l'etichetta non e' riconducibile ad alcuna categoria nota.
    """
    key = "".join(ch for ch in str(raw).lower() if ch.isalnum())
    if key not in _CATEGORY_ALIASES:
        raise ValueError(f"Categoria non riconosciuta: {raw!r}")
    return _CATEGORY_ALIASES[key]


def target_from_category(category: str) -> int:
    """Target binario per la Fase 4: 1 = linfoma follicolare, 0 = tessuto reattivo."""
    return CATEGORY_TARGETS[normalize_category(category)]


def short_label(category: str) -> str:
    """Etichetta breve leggibile ("FL" / "REACTIVE") per output a schermo e grafici."""
    return CATEGORY_SHORT_LABELS[normalize_category(category)]


# ---------------------------------------------------------------------------
# Risoluzione degli input
#
# Principio comune: un input mancante solleva FileNotFoundError invece di
# essere ignorato. Una patch non processata deve essere un errore visibile,
# non una riga assente dal CSV o un overlay che semplicemente non compare
# (audit Fase 3, problema B1).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImageInputs:
    """I due file prodotti dalla Fase 1 per una immagine."""

    stem: str
    category: str
    h_channel_path: Path    # Fase 1 — H-channel CLAHE (segmentazione, tessitura)
    rgb_path: Path          # Fase 1 — RGB normalizzata (overlay, anteprime)


@dataclass(frozen=True)
class PatchInputs:
    """I tre file che descrivono una patch a valle delle Fasi 1 e 2."""

    stem: str
    category: str
    mask_path: Path         # Fase 2 — maschera d'istanza 16-bit
    rgb_path: Path          # Fase 1 — RGB normalizzata (anteprime, overlay)
    h_channel_path: Path    # Fase 1 — H-channel CLAHE (tessitura cromatinica)


def _require(path: Path, stem: str) -> Path:
    """Verifica l'esistenza di un input, con un messaggio d'errore azionabile."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input mancante per l'immagine '{stem}': {path} — "
            "verificare che la Fase 1 sia stata eseguita su tutto il dataset."
        )
    return path


def _require_dir(path: Path, fase: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Directory non trovata: {path} — eseguire prima la {fase}.")
    return path


def iter_h_channel_inputs(category: str, fase1_dir: Path) -> Iterator[ImageInputs]:
    """
    Itera gli output della Fase 1 di una categoria (input della Fase 2).

    Raises:
        FileNotFoundError: se manca la directory degli H-channel o se un
            H-channel non ha la corrispondente RGB normalizzata.
    """
    category = normalize_category(category)
    h_dir = _require_dir(Path(fase1_dir) / category / "h_channel", "Fase 1")
    rgb_dir = Path(fase1_dir) / category / "rgb_normalized"

    for h_channel_path in sorted(h_dir.glob(f"*{H_CHANNEL_SUFFIX}")):
        stem = stem_from_h_channel_name(h_channel_path.name)
        yield ImageInputs(
            stem=stem,
            category=category,
            h_channel_path=h_channel_path,
            rgb_path=_require(rgb_dir / rgb_normalized_name(stem), stem),
        )


def iter_patch_inputs(category: str, fase1_dir: Path, fase2_dir: Path) -> Iterator[PatchInputs]:
    """
    Itera le patch di una categoria restituendo i percorsi dei tre file di input.

    Raises:
        FileNotFoundError: se manca la directory delle maschere o se una
            maschera non ha la corrispondente RGB normalizzata o H-channel.
    """
    category = normalize_category(category)
    mask_dir = _require_dir(Path(fase2_dir) / category / "masks", "Fase 2")
    rgb_dir = Path(fase1_dir) / category / "rgb_normalized"
    h_dir = Path(fase1_dir) / category / "h_channel"

    for mask_path in sorted(mask_dir.glob(f"*{MASK_SUFFIX}")):
        stem = stem_from_mask_name(mask_path.name)
        yield PatchInputs(
            stem=stem,
            category=category,
            mask_path=mask_path,
            rgb_path=_require(rgb_dir / rgb_normalized_name(stem), stem),
            h_channel_path=_require(h_dir / h_channel_name(stem), stem),
        )
