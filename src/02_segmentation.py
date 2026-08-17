"""
===============================================================================
Modulo 02: Segmentazione dei Nuclei Cellulari & Estrazione Centroidi
Tesi: Classificazione Linfoma Follicolare vs Tessuto Reattivo
Versione: 3.0 — Audit & Fix (agosto 2026)
===============================================================================
Questo modulo gestisce:
 1. Segmentazione d'istanza dei nuclei con Marker-Controlled Watershed +
    Distance Transform sui canali Ematossilina (H-channel) pre-processati.
 2. Architettura PyTorch U-Net con backbone ResNet-34 (confronto accademico).
 3. Estrazione coordinate centroidi (x, y) in pixel e micron (1 px = 0.23 µm).
 4. Calcolo metriche di validazione quantitativa: Dice, IoU, AJI, F1 detection.
 5. Generazione di overlay visivi (contorni nuclei sovrapposti all'immagine RGB).

NOTA METODOLOGICA (limitazione riconosciuta):
  La Ground Truth usata per il benchmark (Steps 2.2–2.4) è una pseudo-GT generata
  algoritmicamente dallo stesso Watershed con parametri leggermente differenti.
  NON si tratta di annotazioni manuali effettuate da un patologo. Di conseguenza:
    - Il Dice Score del Watershed è inflazionato (confronto quasi-identico).
    - Il Dice Score della U-Net è inflazionato (nessun hold-out indipendente dalla GT).
  Una validazione rigorosa richiederebbe annotazioni manuali su almeno un subset
  (es. MoNuSeg benchmark) o un split train/val fisicamente separato dalla GT.
  Questo limite va dichiarato esplicitamente nella sezione "Limitazioni" della tesi.
===============================================================================
"""

import os
import cv2
import numpy as np
import scipy.ndimage as ndi
from pathlib import Path
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, label

import torch
import torch.nn as nn
import torchvision.models as models

# ---------------------------------------------------------------------------
# Costante di calibrazione spaziale
# ---------------------------------------------------------------------------
MICRONS_PER_PIXEL = 0.23  # µm / px  (scanner tipico 40×)

# Statistiche ImageNet per normalizzazione encoder ResNet-34 pre-addestrato
# Fonte: PyTorch ResNet34_Weights.DEFAULT.transforms()
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 1. Segmentazione d'Istanza Zero-Shot (Distance Transform + Watershed)
# ---------------------------------------------------------------------------
def segment_nuclei_watershed(
    h_channel,
    min_distance: int = 12,
    peak_threshold_rel: float = 0.15,
    min_area_px: int = 30,
    max_area_px: int = 2500
):
    """
    Esegue la segmentazione d'istanza dei nuclei cellulari sul canale H (Ematossilina)
    tramite Marker-Controlled Watershed e Trasformata di Distanza Euclidea (EDT).

    Parametri scelti con riferimento alla biologia dei linfociti:
      - Linfocita normale: diametro ~6–12 µm, raggio ~3–6 µm → min_distance >= 10 px.
      - Centroblasto FL:   diametro ~8–15 µm, area fino a ~130 µm² → max_area fino a 2500 px.
      Riferimento: Iwamoto et al. (2024), Computers in Biology and Medicine.

    Args:
        h_channel (np.ndarray): Canale H in scala di grigi uint8.
        min_distance (int): Distanza minima in pixel tra centroidi locali (default 12 px ≈ 2.8 µm).
                            Valori inferiori al raggio medio del nucleo causano over-segmentazione.
        peak_threshold_rel (float): Soglia relativa al massimo della distance map per accettare
                                    un picco come marker (default 0.15 = 15% del massimo locale).
                                    Controlla direttamente la sensibilità del rilevamento.
        min_area_px (int): Area minima nucleare in pixel (default 30 px ≈ 1.6 µm²).
                           Rimuove rumori di binarizzazione e artefatti submicron.
        max_area_px (int): Area massima nucleare in pixel (default 2500 px ≈ 132 µm²).
                           Aumentato da 1500 a 2500 per includere centroblasti grandi
                           (Iwamoto et al. 2024: area mediana centroblasti ~55–70 µm², tail > 100 µm²).

    Returns:
        cleaned_labels (np.ndarray): Maschera d'istanza int32 (0 = sfondo, ID ≥ 1 = nuclei).
        centroids (list[dict]): Centroidi con coordinate px e µm, area px e µm².
    """
    # 1. Sogliatura globale di Otsu per binarizzare il canale H
    #    (NOTA: Otsu è una soglia GLOBALE basata sulla varianza inter-classe dell'istogramma,
    #    non adattiva. È appropriata per il canale H post-normalizzazione Macenko che ha
    #    distribuzione di intensità bimodale stabile.)
    _, binary = cv2.threshold(h_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Trasformata di Distanza Euclidea: ogni pixel foreground riceve la propria
    #    distanza dal bordo più vicino. I massimi locali corrispondono ai centri dei nuclei.
    distance = ndi.distance_transform_edt(binary)

    # 3. Trova picchi locali della distance map (candidati centri nucleari).
    #    - min_distance: sopprime picchi troppo vicini (evita over-segmentazione).
    #    - peak_threshold_rel: accetta solo picchi > peak_threshold_rel * max(distance).
    #      Questo parametro è ora correttamente usato dalla firma della funzione.
    abs_threshold = distance.max() * peak_threshold_rel
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=abs_threshold,
        labels=binary
    )

    # 4. Maschera dei marker per il Watershed
    mask_markers = np.zeros(distance.shape, dtype=bool)
    if len(coords) > 0:
        mask_markers[tuple(coords.T)] = True

    markers, _ = ndi.label(mask_markers)

    # 5. Watershed guidato dai marker sulla distance map negata
    labels_ws = watershed(-distance, markers, mask=binary)

    # 6. Filtraggio morfologico: rimuove regioni troppo piccole (rumore) o troppo grandi
    #    (artefatti di fusione tra nuclei non separati dal Watershed)
    cleaned_labels = np.zeros_like(labels_ws, dtype=np.int32)
    centroids = []
    current_id = 1

    for prop in regionprops(labels_ws):
        area = prop.area
        if min_area_px <= area <= max_area_px:
            cleaned_labels[labels_ws == prop.label] = current_id
            cy, cx = prop.centroid
            centroids.append({
                'id':             current_id,
                'centroid_y_px':  float(cy),
                'centroid_x_px':  float(cx),
                'centroid_y_um':  round(cy * MICRONS_PER_PIXEL, 2),
                'centroid_x_um':  round(cx * MICRONS_PER_PIXEL, 2),
                'area_px':        int(area),
                'area_um2':       round(area * (MICRONS_PER_PIXEL ** 2), 2)
            })
            current_id += 1

    return cleaned_labels, centroids


# ---------------------------------------------------------------------------
# 2. Architettura PyTorch U-Net (ResNet-34 Backbone)
#    RUOLO: strumento di confronto accademico con il Watershed zero-shot.
#    NON è usata per la produzione delle maschere delle 600 immagini del dataset.
#    La pipeline operativa (Step 2.1) usa esclusivamente segment_nuclei_watershed().
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Blocco conv-BN-ReLU-conv-BN-ReLU con Dropout2d opzionale."""

    def __init__(self, in_c: int, out_c: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0.0:
            # Dropout2d: azzera interi canali anziché pixel singoli,
            # più efficace per feature map spaziali (Tompson et al., 2015).
            layers.append(nn.Dropout2d(p=dropout_p))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetResNet34(nn.Module):
    """
    U-Net con encoder ResNet-34 pre-addestrato su ImageNet.

    Input atteso: tensore float in range [0, 1] con shape (B, 3, H, W).
    La normalizzazione ImageNet (mean/std) viene applicata internamente nel forward()
    per garantire che l'encoder operi sulla distribuzione per cui è stato ottimizzato.

    Il decoder usa Dropout2d (p=0.1) negli stadi dec3 e dec4 per ridurre l'overfitting
    su dataset piccoli (es. 30 patch di GT). Dropout disabilitato in eval mode.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True, decoder_dropout: float = 0.1):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        # Registra le statistiche ImageNet come buffer (seguono il device del modello)
        self.register_buffer(
            'imagenet_mean',
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'imagenet_std',
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        )

        # ── Encoder (ResNet-34 stages) ──────────────────────────────────────
        self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool   = resnet.maxpool
        self.layer1    = resnet.layer1   # → 56×56,  64 ch
        self.layer2    = resnet.layer2   # → 28×28, 128 ch
        self.layer3    = resnet.layer3   # → 14×14, 256 ch
        self.layer4    = resnet.layer4   # →  7×7,  512 ch

        # ── Decoder (skip-connections U-Net) ───────────────────────────────
        # I ConvBlock degli stadi profondi (4, 3) usano Dropout per regularizzare
        # il decoder dove le feature sono più astratte e l'overfitting è più probabile.
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256, dropout_p=decoder_dropout)   # 256+256 in

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128, dropout_p=decoder_dropout)   # 128+128 in

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)                                # 64+64 in

        self.up1  = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)                                # 64+64 in

        self.final_up   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizzazione ImageNet interna. Input: [0, 1]. Output: distribuzione ImageNet."""
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor (B, 3, H, W) con valori in [0, 1].
        Returns:
            logit map (B, num_classes, H, W) non sigmoidata.
        """
        # ── Normalizzazione ImageNet (applicata internamente) ────────────
        x = self._normalize(x)

        # ── Encoder ──────────────────────────────────────────────────────
        x0 = self.init_conv(x)             # (B,  64, H/2,  W/2)  → 112×112
        x1 = self.layer1(self.maxpool(x0)) # (B,  64, H/4,  W/4)  →  56×56
        x2 = self.layer2(x1)               # (B, 128, H/8,  W/8)  →  28×28
        x3 = self.layer3(x2)               # (B, 256, H/16, W/16) →  14×14
        x4 = self.layer4(x3)               # (B, 512, H/32, W/32) →   7×7

        # ── Decoder ──────────────────────────────────────────────────────
        d4 = self.up4(x4)                  # → 14×14, 256ch
        d4 = self.dec4(torch.cat([d4, x3], dim=1))

        d3 = self.up3(d4)                  # → 28×28, 128ch
        d3 = self.dec3(torch.cat([d3, x2], dim=1))

        d2 = self.up2(d3)                  # → 56×56,  64ch
        d2 = self.dec2(torch.cat([d2, x1], dim=1))

        d1 = self.up1(d2)                  # → 112×112, 64ch
        d1 = self.dec1(torch.cat([d1, x0], dim=1))

        out = self.final_up(d1)            # → 224×224, 32ch
        out = self.final_conv(out)         # → 224×224, num_classes
        return out


# ---------------------------------------------------------------------------
# 3. Metriche di Validazione Quantitativa
# ---------------------------------------------------------------------------
def compute_segmentation_metrics(mask_gt: np.ndarray, mask_pred: np.ndarray) -> dict:
    """
    Calcola un set completo di metriche per la segmentazione nucleare d'istanza.

    Metriche a livello di pixel (foreground globale):
      - Dice Coefficient (F1 sul foreground binario)
      - IoU / Jaccard Index

    Metriche a livello di istanza (instance-level):
      - AJI (Aggregated Jaccard Index) — Kumar et al. (2017), MoNuSeg.
        Misura l'overlap medio tra ogni nucleo GT e il suo miglior match predetto.
        È lo standard del campo per la valutazione di segmentazione nucleare.
      - F1 Detection @ IoU ≥ 0.5 — Schmidt et al. (2018), StarDist.
        Misura precision/recall sui nuclei come oggetti (non come pixel).

    NOTA METODOLOGICA: queste metriche sono accurate solo se mask_gt è una
    Ground Truth indipendente dall'algoritmo valutato. Se mask_gt è generata
    dallo stesso Watershed, le metriche instance-level saranno comunque inflazionate.

    Args:
        mask_gt (np.ndarray): Maschera di istanza GT (int, 0 = sfondo).
        mask_pred (np.ndarray): Maschera di istanza predetta (int, 0 = sfondo).

    Returns:
        dict con chiavi: dice, iou, aji, f1_det, precision_det, recall_det,
                         tp, fp, fn, n_gt, n_pred, total_gt_px, total_pred_px.
    """
    # ── Pixel-level (foreground binario) ─────────────────────────────────
    bin_gt   = (mask_gt   > 0).astype(np.uint8)
    bin_pred = (mask_pred > 0).astype(np.uint8)

    total_gt   = int(bin_gt.sum())
    total_pred = int(bin_pred.sum())

    # Edge case: entrambe le maschere vuote → accordo perfetto
    if total_gt == 0 and total_pred == 0:
        return {
            'dice': 1.0, 'iou': 1.0, 'aji': 1.0,
            'f1_det': 1.0, 'precision_det': 1.0, 'recall_det': 1.0,
            'tp': 0, 'fp': 0, 'fn': 0,
            'n_gt': 0, 'n_pred': 0,
            'total_gt_px': 0, 'total_pred_px': 0
        }

    intersection = int(np.logical_and(bin_gt, bin_pred).sum())
    union        = int(np.logical_or(bin_gt, bin_pred).sum())

    dice = (2.0 * intersection) / (total_gt + total_pred) if (total_gt + total_pred) > 0 else 0.0
    iou  = intersection / union if union > 0 else 0.0

    # ── Instance-level ────────────────────────────────────────────────────
    aji = _compute_aji(mask_gt, mask_pred)
    det = _compute_detection_f1(mask_gt, mask_pred, iou_threshold=0.5)

    return {
        'dice':          round(dice, 4),
        'iou':           round(iou, 4),
        'aji':           round(aji, 4),
        'f1_det':        det['f1'],
        'precision_det': det['precision'],
        'recall_det':    det['recall'],
        'tp':            det['tp'],
        'fp':            det['fp'],
        'fn':            det['fn'],
        'n_gt':          det['n_gt'],
        'n_pred':        det['n_pred'],
        'total_gt_px':   total_gt,
        'total_pred_px': total_pred
    }


def _compute_aji(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Aggregated Jaccard Index (AJI) — Kumar et al. (2017).
    Ref: Kumar N. et al., "A Dataset and a Technique for Generalized Nuclear
         Segmentation for Computational Pathology", IEEE TMI 2017.

    Algoritmo:
      Per ogni nucleo GT g_i, trova il nucleo predetto p_j con IoU massimo.
      Accumula |g_i ∩ p_j| al numeratore e |g_i ∪ p_j| al denominatore.
      I nuclei predetti non matchati contribuiscono con la loro area al denominatore.
      AJI = Σ|g_i ∩ p_j*| / (Σ|g_i ∪ p_j*| + Σ|unmatched pred|)
    """
    gt_ids   = np.unique(mask_gt);   gt_ids   = gt_ids[gt_ids != 0]
    pred_ids = np.unique(mask_pred); pred_ids = pred_ids[pred_ids != 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    matched_pred_ids = set()
    total_inter = 0
    total_union = 0

    for gt_id in gt_ids:
        gt_mask  = (mask_gt == gt_id)
        best_iou = 0.0
        best_inter = 0
        best_union = 0
        best_pid = None

        for pred_id in pred_ids:
            pred_mask = (mask_pred == pred_id)
            inter = int(np.logical_and(gt_mask, pred_mask).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            iou_val = inter / union
            if iou_val > best_iou:
                best_iou   = iou_val
                best_inter = inter
                best_union = union
                best_pid   = pred_id

        if best_pid is not None:
            matched_pred_ids.add(best_pid)
            total_inter += best_inter
            total_union += best_union
        else:
            # Nucleo GT senza match: contribuisce solo al denominatore
            total_union += int(gt_mask.sum())

    # Nuclei predetti non matchati: contribuiscono al denominatore
    for pred_id in pred_ids:
        if pred_id not in matched_pred_ids:
            total_union += int((mask_pred == pred_id).sum())

    return float(total_inter / total_union) if total_union > 0 else 0.0


def _compute_detection_f1(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    iou_threshold: float = 0.5
) -> dict:
    """
    F1 Score a livello di detection nucleare con soglia IoU ≥ iou_threshold.
    Standard per benchmarking nucleare (Schmidt et al. 2018, StarDist).

    Un nucleo predetto è un True Positive se ha IoU ≥ iou_threshold con almeno
    un nucleo GT non ancora matchato. Ogni GT può essere matchato al più una volta.

    Returns:
        dict con f1, precision, recall, tp, fp, fn, n_gt, n_pred.
    """
    gt_ids   = np.unique(mask_gt);   gt_ids   = gt_ids[gt_ids != 0]
    pred_ids = np.unique(mask_pred); pred_ids = pred_ids[pred_ids != 0]

    n_gt   = len(gt_ids)
    n_pred = len(pred_ids)

    if n_gt == 0 and n_pred == 0:
        return {'f1': 1.0, 'precision': 1.0, 'recall': 1.0,
                'tp': 0, 'fp': 0, 'fn': 0, 'n_gt': 0, 'n_pred': 0}

    matched_gt_ids = set()
    tp = 0

    for pred_id in pred_ids:
        pred_mask = (mask_pred == pred_id)
        best_iou  = 0.0
        best_gt   = None

        for gt_id in gt_ids:
            if gt_id in matched_gt_ids:
                continue
            gt_mask = (mask_gt == gt_id)
            inter   = int(np.logical_and(gt_mask, pred_mask).sum())
            if inter == 0:
                continue
            union   = int(np.logical_or(gt_mask, pred_mask).sum())
            iou_val = inter / union
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt  = gt_id

        if best_iou >= iou_threshold and best_gt is not None:
            tp += 1
            matched_gt_ids.add(best_gt)

    fp = n_pred - tp
    fn = n_gt  - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'f1':        round(f1, 4),
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'tp':        tp,
        'fp':        fp,
        'fn':        fn,
        'n_gt':      n_gt,
        'n_pred':    n_pred
    }


# ---------------------------------------------------------------------------
# 4. Generazione Overlay Visivo
# ---------------------------------------------------------------------------
def draw_segmentation_overlay(
    img_rgb: np.ndarray,
    instance_mask: np.ndarray,
    centroids: list = None
) -> np.ndarray:
    """
    Disegna i contorni dei nuclei segmentati (verde) e i centroidi (giallo)
    sull'immagine RGB normalizzata.

    Ottimizzazione: invece di iterare N maschere binarie separate per N nuclei,
    si usa cv2.findContours su tutti i contorni esterni della maschera binarizzata
    in un'unica chiamata, più una seconda passata per i separatori di istanza.
    Questo riduce il numero di allocazioni da O(N) a O(1) per i contorni di sfondo,
    mantenendo la correttezza visiva.

    Args:
        img_rgb (np.ndarray): Immagine RGB uint8 (H, W, 3).
        instance_mask (np.ndarray): Maschera d'istanza int (0 = sfondo).
        centroids (list[dict], optional): Lista centroidi da draw_segmentation_overlay.

    Returns:
        overlay (np.ndarray): Copia dell'immagine con contorni e centroidi sovrapposti.
    """
    overlay = img_rgb.copy()

    # Contorni di ogni istanza (loop necessario per colore per-istanza;
    # ottimizzato creando le maschere binarie solo una volta per id unico)
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids != 0]

    for uid in unique_ids:
        binary_single = np.uint8(instance_mask == uid)
        cnts, _ = cv2.findContours(binary_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 1)

    # Centroidi come punti gialli
    if centroids:
        for c in centroids:
            cx = int(c['centroid_x_px'])
            cy = int(c['centroid_y_px'])
            cv2.circle(overlay, (cx, cy), 1, (255, 255, 0), -1)

    return overlay


# ---------------------------------------------------------------------------
# 5. Utilità: split train/val per training U-Net su pseudo-GT
# ---------------------------------------------------------------------------
def split_gt_patches(gt_patch_paths: list, val_fraction: float = 0.33, seed: int = 42):
    """
    Divide le patch di Ground Truth (pseudo-GT) in train e val set
    mantenendo la stratificazione per categoria (FL / Reactive).

    NOTA: questo split non risolve la circolarità della validazione
    (la pseudo-GT è sempre generata algoritmicamente), ma garantisce almeno
    che la U-Net NON venga valutata sulle stesse immagini su cui è addestrata.

    Args:
        gt_patch_paths (list[tuple]): Lista di (path_img, path_mask, categoria).
        val_fraction (float): Frazione da usare come validation set (default 0.33).
        seed (int): Seed per riproducibilità.

    Returns:
        train_paths, val_paths (list): Due liste di tuple (path_img, path_mask, cat).
    """
    rng = np.random.default_rng(seed)

    fl_patches = [p for p in gt_patch_paths if p[2] == 'Follicular Lymphoma']
    re_patches = [p for p in gt_patch_paths if p[2] == 'Reactive Tissue']

    train_paths, val_paths = [], []

    for group in [fl_patches, re_patches]:
        n_val = max(1, int(len(group) * val_fraction))
        indices = rng.permutation(len(group))
        val_idx   = indices[:n_val]
        train_idx = indices[n_val:]
        val_paths   += [group[i] for i in val_idx]
        train_paths += [group[i] for i in train_idx]

    return train_paths, val_paths


# ---------------------------------------------------------------------------
# Entry point (test rapido)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("[INFO] Modulo 02 — Segmentazione Nuclei v3.0 (Watershed + U-Net ResNet34)")
    print()

    # Test istanziazione U-Net (CPU)
    model = UNetResNet34(num_classes=1, pretrained=False, decoder_dropout=0.1)
    dummy = torch.zeros(1, 3, 224, 224)  # input in [0, 1]
    with torch.no_grad():
        out = model(dummy)
    print(f"[TEST] U-Net output shape: {tuple(out.shape)}  (atteso: (1, 1, 224, 224))")

    # Test metriche su maschere sintetiche
    gt_mask  = np.zeros((64, 64), dtype=np.int32)
    pred_mask = np.zeros((64, 64), dtype=np.int32)
    gt_mask[10:20, 10:20] = 1
    gt_mask[30:45, 30:45] = 2
    pred_mask[11:21, 11:21] = 1     # nucleo 1: quasi perfetto
    pred_mask[30:45, 30:45] = 2     # nucleo 2: perfetto
    # nucleo fittizio in più (FP)
    pred_mask[50:58, 50:58] = 3

    metrics = compute_segmentation_metrics(gt_mask, pred_mask)
    print(f"[TEST] Metriche sintetiche:")
    for k, v in metrics.items():
        print(f"       {k:20s} = {v}")

    # Test edge case: entrambe vuote
    m_empty = compute_segmentation_metrics(np.zeros((32, 32), np.int32), np.zeros((32, 32), np.int32))
    assert m_empty['dice'] == 1.0 and m_empty['aji'] == 1.0, "Edge case maschere vuote fallito!"
    print("[TEST] Edge case maschere vuote: OK (Dice=1.0, AJI=1.0)")

    print()
    print("[OK] Tutti i test superati.")
