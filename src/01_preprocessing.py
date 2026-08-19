"""
===============================================================================
Modulo 01: Preprocessing delle Immagini Istologiche (H&E)
Tesi: Classificazione Linfoma Follicolare vs Tessuto Reattivo
===============================================================================
Questo modulo gestisce:
 1. Calibrazione spaziale (1 px = 0.23 µm)
 2. Stain Normalization H&E con Macenko (SVD in Densità Ottica)
    - Selezione automatica della Reference Image più rappresentativa del dataset
 3. Denoising con Filtro Bilaterale (bordi nucleari preservati)
 4. Color Deconvolution (Ruifrok & Johnston) + CLAHE
    per estrarre il Canale Ematossilina (H-channel) ad alto contrasto

CORREZIONI APPLICATE (v2):
 - FIX 1: rgb_normalized ora salva img_denoised (stesso stadio di h_channel)
 - FIX 2: Reference Image selezionata automaticamente come patch più vicina
          alla mediana cromatica dell'intero dataset (non più la prima alfabetica)
 - FIX 3: Logica SVD estratta in metodo privato _estimate_HE_vectors() (DRY)
 - FIX 4: Rimosso import inutile di matplotlib
 - AGGIUNTA: Salvataggio preprocessing_metadata.json per riproducibilità

CORREZIONI APPLICATE (v3, 19/08/2026 — code review):
 - FIX 5: _estimate_HE_vectors() ora stima gli estremi H/E tramite atan2(angolo) +
          percentile sull'angolo, invece del rapporto Phi[:,1]/Phi[:,0]. Le due
          formulazioni sono state verificate numericamente equivalenti su dati OD
          H&E realistici, ma si adotta la forma con atan2 per aderenza letterale
          al metodo pubblicato da Macenko et al. (2009) — citabilità diretta in
          tesi — e per eliminare il rischio residuo di instabilità del rapporto
          quando la proiezione sul primo asse principale è vicina a zero.

Reference:
 - Macenko et al. (2009) IEEE ISBI
 - Ruifrok & Johnston (2001) Anal. Quant. Cytol. Histol.
===============================================================================
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Costante di calibrazione spaziale
# Scanner: Hamamatsu NanoZoomer S360, Obiettivo 40x
# ---------------------------------------------------------------------------
MICRONS_PER_PIXEL = 0.23          # µm / px
PIXEL_AREA_UM2    = MICRONS_PER_PIXEL ** 2  # µm² / px²
PATCH_SIZE_PX     = 224
PATCH_SIZE_UM     = PATCH_SIZE_PX * MICRONS_PER_PIXEL  # 51.52 µm
PATCH_AREA_UM2    = PATCH_SIZE_UM ** 2                  # 2654.31 µm²


# ---------------------------------------------------------------------------
# Selezione automatica della Reference Image (FIX 2)
# ---------------------------------------------------------------------------
def find_best_reference_image(image_dirs):
    """
    Seleziona la patch istologica più rappresentativa del dataset come
    immagine di riferimento (Target Reference) per la Normalizzazione di Macenko.

    Strategia: calcola la media BGR di ogni immagine, poi restituisce il path
    della patch con media più vicina (distanza euclidea minima) alla mediana
    dell'intero dataset. Questo evita di usare arbitrariamente la prima immagine
    ordinata alfabeticamente, che potrebbe essere cromaticamente atipica.

    Args:
        image_dirs (list[str]): Lista di directory contenenti le immagini raw.

    Returns:
        str: Path assoluto della best reference image.
        dict: Metadati della selezione (path, distanza dalla mediana, ecc.).
    """
    all_means = []
    all_paths = []

    for d in image_dirs:
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                p = os.path.join(d, f)
                img = cv2.imread(p)
                if img is not None:
                    all_means.append(img.mean(axis=(0, 1)))
                    all_paths.append(p)

    if not all_means:
        raise ValueError("Nessuna immagine trovata nelle directory specificate.")

    all_means = np.array(all_means)
    dataset_median = np.median(all_means, axis=0)
    distances = np.linalg.norm(all_means - dataset_median, axis=1)
    best_idx = int(np.argmin(distances))

    metadata = {
        "reference_image": all_paths[best_idx],
        "reference_image_basename": os.path.basename(all_paths[best_idx]),
        "distance_from_dataset_median": float(distances[best_idx]),
        "dataset_median_bgr": dataset_median.tolist(),
        "n_images_evaluated": len(all_paths)
    }

    return all_paths[best_idx], metadata


# ---------------------------------------------------------------------------
# Stain Normalizer di Macenko (FIX 3: SVD estratta in metodo privato)
# ---------------------------------------------------------------------------
class StainNormalizerMacenko:
    """
    Normalizzazione cromatica H&E con il metodo di Macenko (SVD in OD space).

    Reference: Macenko et al. (2009) "A method for normalizing histology slides
    for quantitative analysis", IEEE ISBI, pp. 1107–1110.
    DOI: 10.1109/ISBI.2009.5193250

    Parametri:
        Io (int):    Intensità massima attesa (255 per immagini uint8).
        alpha (float): Percentile per la stima dei vettori H/E (default 1%).
        beta (float):  Soglia OD per filtrare i pixel trasparenti (default 0.15).
    """

    def __init__(self, Io=255, alpha=1, beta=0.15):
        self.Io    = Io
        self.alpha = alpha
        self.beta  = beta
        # Valori di default (Macenko 2009 paper) — sovrascritti da fit()
        self.HERef   = np.array([[0.5626, 0.2159],
                                  [0.7201, 0.8012],
                                  [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    # -- Metodo privato: logica SVD condivisa tra fit() e transform() --
    def _estimate_HE_vectors(self, OD_flat):
        """
        Stima i vettori di assorbimento dell'Ematossilina e dell'Eosina
        tramite SVD applicata ai pixel ad alta densità ottica.

        Args:
            OD_flat (np.ndarray): Array (N, 3) di valori OD dei pixel.

        Returns:
            np.ndarray: Matrice (3, 2) con i vettori H ed E normalizzati.
                        Colonna 0 = Ematossilina, Colonna 1 = Eosina.
        """
        # Filtra pixel trasparenti (OD bassa = pixel chiari = sfondo vuoto)
        ODhat = OD_flat[np.all(OD_flat >= self.beta, axis=1)]
        if len(ODhat) == 0:
            return self.HERef  # fallback ai valori di default

        # SVD per trovare il piano principale dell'assorbimento
        _, _, V = np.linalg.svd(ODhat, full_matrices=False)
        V = V[:2, :]

        # Garantisce orientamento consistente dei vettori
        if V[0, 0] < 0:
            V[0, :] *= -1
        if V[1, 0] < 0:
            V[1, :] *= -1

        # Proietta i pixel sul piano e stima i due estremi (H e E) tramite l'angolo
        # polare di ciascun pixel proiettato — formulazione standard di Macenko et al.
        # (2009): atan2(y, x) + percentile sull'angolo, come nella quasi totalita' delle
        # implementazioni di riferimento pubbliche del metodo.
        # NOTA (v3, 19/08/2026): versione precedente usava il rapporto Phi[:,1]/Phi[:,0]
        # anziche' l'angolo. Verificato numericamente (simulazione su dati OD H&E
        # realistici) che le due formulazioni sono equivalenti quando la proiezione sul
        # primo asse principale (Phi[:,0]) resta positiva, come tipicamente accade per
        # dati di assorbanza ottica fisiologici. Si adotta comunque atan2 per aderenza
        # letterale al metodo pubblicato (citabilita' diretta di Macenko et al. 2009 in
        # tesi) e per eliminare il rischio residuo di instabilita' del rapporto quando
        # Phi[:,0] si avvicina a zero (patch con tessuto molto scarso).
        Phi = np.dot(ODhat, V.T)
        phi = np.arctan2(Phi[:, 1], Phi[:, 0])
        phi_min = np.percentile(phi, self.alpha)
        phi_max = np.percentile(phi, 100 - self.alpha)

        vMin = np.dot(V.T, np.array([np.cos(phi_min), np.sin(phi_min)]))
        vMax = np.dot(V.T, np.array([np.cos(phi_max), np.sin(phi_max)]))

        # Convenzione: colonna 0 = vettore con componente R maggiore (Ematossilina)
        if vMin[0] > vMax[0]:
            HE = np.column_stack((vMin, vMax))
        else:
            HE = np.column_stack((vMax, vMin))

        return HE / np.linalg.norm(HE, axis=0)

    def fit(self, img_rgb):
        """
        Stima i vettori H/E e la normalizzazione della concentrazione
        dalla patch di riferimento (Target Reference).

        Deve essere chiamato UNA SOLA VOLTA sulla reference image prima
        di chiamare transform() su tutte le altre immagini.
        """
        img_flat = img_rgb.astype(np.float64).reshape((-1, 3))
        OD       = -np.log10((img_flat + 1.0) / (self.Io + 1.0))  # +1/+1 garantisce OD ≥ 0

        self.HERef = self._estimate_HE_vectors(OD)

        # Stima la concentrazione massima (99° percentile) nella reference
        ODhat  = OD[np.all(OD >= self.beta, axis=1)]
        if len(ODhat) > 0:
            C            = np.linalg.lstsq(self.HERef, ODhat.T, rcond=None)[0]
            self.maxCRef = np.percentile(C, 99, axis=1)

    def transform(self, img_rgb):
        """
        Normalizza i colori H&E di img_rgb trascinandoli verso la distribuzione
        della reference image su cui è stato chiamato fit().

        Args:
            img_rgb (np.ndarray): Immagine H&W×3 in formato RGB uint8.

        Returns:
            np.ndarray: Immagine normalizzata RGB uint8 della stessa dimensione.
        """
        h, w, _ = img_rgb.shape
        img_flat = img_rgb.astype(np.float64).reshape((-1, 3))
        OD       = -np.log10((img_flat + 1.0) / (self.Io + 1.0))  # +1/+1 garantisce OD ≥ 0

        # Stima vettori H/E della patch sorgente
        HE_src = self._estimate_HE_vectors(OD)

        # Calcola le concentrazioni di H ed E in ogni pixel sorgente
        C    = np.linalg.lstsq(HE_src, OD.T, rcond=None)[0]
        maxC = np.percentile(C, 99, axis=1)
        maxC[maxC == 0] = 1e-5

        # Riscala le concentrazioni verso quelle della reference
        C_norm = (C / maxC[:, None]) * self.maxCRef[:, None]

        # Ricostruisce l'immagine normalizzata dallo spazio OD
        OD_norm  = np.dot(self.HERef, C_norm)
        img_norm = self.Io * (10 ** (-OD_norm))
        img_norm = np.clip(img_norm.T, 0, 255).astype(np.uint8)

        return img_norm.reshape((h, w, 3))

    def get_params(self):
        """Restituisce i parametri correnti per il salvataggio nei metadati."""
        return {
            "Io": self.Io,
            "alpha": self.alpha,
            "beta": self.beta,
            "HERef": self.HERef.tolist(),
            "maxCRef": self.maxCRef.tolist()
        }


# ---------------------------------------------------------------------------
# Denoising con Filtro Bilaterale
# ---------------------------------------------------------------------------
def apply_bilateral_denoising(img_rgb, d=9, sigma_color=75, sigma_space=75):
    """
    Applica il Filtro Bilaterale all'immagine RGB.

    Rispetto al filtro Gaussiano, il filtro bilaterale preserva i contorni
    acuti delle membrane nucleari riducendo il rumore solo nelle aree uniformi.

    Args:
        img_rgb:      Immagine RGB uint8.
        d:            Diametro del vicinato in pixel (default 9).
        sigma_color:  Ampiezza del kernel nello spazio del colore (default 75).
        sigma_space:  Ampiezza del kernel nello spazio geometrico (default 75).

    Returns:
        np.ndarray: Immagine RGB filtrata.
    """
    return cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)


# ---------------------------------------------------------------------------
# Color Deconvolution (Ruifrok & Johnston) + CLAHE
# ---------------------------------------------------------------------------
def extract_hematoxylin_channel_clahe(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Estrae il Canale Ematossilina (H-channel) dall'immagine H&E tramite
    Color Deconvolution (Ruifrok & Johnston, 2001) e ne aumenta il contrasto
    con CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Il risultato è una mappa in scala di grigi ad alto contrasto in cui i
    nuclei cellulari (ricchi di Ematossilina) appaiono come picchi brillanti
    su un fondo scuro — formato ideale per la segmentazione con StarDist/U-Net.

    Calibrazione: 1 tile CLAHE = (224/8)×(224/8) px = 28×28 px ≈ 6.4×6.4 µm²
    (scala confrontabile con la dimensione di un nucleo linfocitario: 5–10 µm).

    Reference: Ruifrok AC & Johnston DA (2001). Anal. Quant. Cytol. Histol. 23(4):291-9.

    Args:
        img_rgb:        Immagine RGB uint8 (già denoisata).
        clip_limit:     Limite di amplificazione CLAHE (default 2.0).
        tile_grid_size: Dimensione della griglia adattiva CLAHE (default 8×8).

    Returns:
        np.ndarray: H-channel in scala di grigi uint8 [0, 255].
    """
    # Matrice di deconvoluzione H&E (Ruifrok & Johnston, 2001)
    # Righe: [Ematossilina, Eosina, Residuo]  |  Colonne: [R, G, B]
    HE_matrix = np.array([
        [0.650, 0.704, 0.286],  # Hematoxylin
        [0.072, 0.990, 0.105],  # Eosin
        [0.268, 0.570, 0.776]   # Residual
    ])
    # Normalizza ogni riga a vettore unitario
    norms = np.linalg.norm(HE_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-5
    HE_matrix = HE_matrix / norms

    # Conversione RGB → Densità Ottica (legge di Beer-Lambert)
    img_float = img_rgb.astype(np.float64) + 1.0       # +1 evita log(0)
    OD = -np.log10(img_float / 256.0)                  # /256 garantisce OD ≥ 0 anche per pixel=255

    # Deconvoluzione: stima le concentrazioni H, E, Residuo per ogni pixel
    OD_flat = OD.reshape((-1, 3))
    stains  = np.dot(OD_flat, np.linalg.pinv(HE_matrix))

    # Estrai il canale Ematossilina (concentrazione H per ogni pixel)
    H_channel = stains[:, 0].reshape(img_rgb.shape[:2])
    H_norm    = cv2.normalize(H_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE per esaltare il contrasto locale della cromatina nucleare
    clahe      = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    H_enhanced = clahe.apply(H_norm)

    return H_enhanced


# ---------------------------------------------------------------------------
# Pipeline Completa su Singola Immagine
# ---------------------------------------------------------------------------
def process_single_image(img_path, normalizer=None):
    """
    Esegue la pipeline completa di preprocessing su una singola immagine.

    Ordine degli step:
      1. Caricamento e conversione BGR→RGB
      2. Normalizzazione cromatica di Macenko (se normalizer non è None)
      3. Denoising bilaterale
      4. Estrazione H-channel con Deconvoluzione di Ruifrok + CLAHE

    IMPORTANTE (FIX 1): sia 'denoised_rgb' che 'hematoxylin_h' derivano
    dallo stesso stadio (img_denoised), garantendo coerenza tra le due
    immagini salvate su disco.

    Args:
        img_path (str | Path): Path dell'immagine da processare.
        normalizer (StainNormalizerMacenko | None): Normalizzatore già fittato.
            Se None, salta la normalizzazione cromatica.

    Returns:
        dict con chiavi:
            'original_rgb'   — immagine grezza RGB (solo lettura)
            'normalized_rgb' — dopo Macenko (prima del denoising)
            'denoised_rgb'   — dopo Macenko + Bilateral Filter  ← STADIO SALVATO
            'hematoxylin_h'  — H-channel monocromatico CLAHE   ← STADIO SALVATO
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Impossibile caricare l'immagine: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: Normalizzazione cromatica
    img_norm = normalizer.transform(img_rgb) if normalizer is not None else img_rgb.copy()

    # Step 2: Denoising bilaterale
    img_denoised = apply_bilateral_denoising(img_norm)

    # Step 3: Estrazione H-channel con CLAHE (applicata su img_denoised)
    h_channel = extract_hematoxylin_channel_clahe(img_denoised)

    return {
        'original_rgb':  img_rgb,
        'normalized_rgb': img_norm,
        'denoised_rgb':  img_denoised,   # ← stesso stadio di h_channel
        'hematoxylin_h': h_channel
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"[INFO] Modulo Preprocessing v2 inizializzato.")
    print(f"[INFO] Calibrazione: {MICRONS_PER_PIXEL} µm/px | "
          f"FOV per patch: {PATCH_SIZE_UM:.2f} x {PATCH_SIZE_UM:.2f} µm")
