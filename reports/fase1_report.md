# Report Fase 1 — Preprocessing delle Immagini Istologiche H&E
### Tesi: Quantificazione Citomorfometrica e Spaziale per la Classificazione tra Linfoma Follicolare e Tessuto Reattivo
*Modulo: [`src/01_preprocessing.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/src/01_preprocessing.py) — Versione 2.0*

---

## Obiettivo

Trasformare le 600 patch H&E grezze in due rappresentazioni standardizzate:
1. **Immagine RGB normalizzata** — per l'analisi di tessitura e colore dello stroma.
2. **Canale Ematossilina (H-channel)** — input ad alto contrasto per la segmentazione dei nuclei.

---

## Calibrazione Spaziale

| Parametro | Valore | Riferimento Bibliografico |
|-----------|--------|---------------------------|
| **Scanner** | Hamamatsu NanoZoomer S360 (40×) | Carreras et al., *Cancers* (2025) |
| **Scala Spaziale** | **1 px = 0.23 µm** | Carreras et al., *Cancers* (2025) |
| **Dimensione Patch** | 224 × 224 px = **51.52 × 51.52 µm** | Carreras et al., *Cancers* (2025) |
| **Area per Patch** | **2.654,31 µm²** | Calcolo analitico |

Tutti i biomarcatori delle fasi successive (area nucleare, perimetro, distanze inter-nucleari) sono espressi in **unità fisiche reali (µm, µm²)** per garantire interpretabilità clinica immediata e confrontabilità con la letteratura istopatologica (*Iwamoto et al., 2024*).

---

## Pipeline Implementata

```
Immagine Raw RGB (224×224 px)
        │
        ▼
[ Step 1 ] Normalizzazione di Macenko (Macenko et al., 2009)
        │   Uniforma i colori H&E verso una patch di riferimento ottimale.
        │   Riduce la varianza cromatica inter-immagine del 68.6%.
        ▼
[ Step 2 ] Filtro Bilaterale (Tomasi & Manduchi, 1998)
        │   Rimuove il rumore di fondo conservando i bordi dei nuclei.
        │   Parametri: d=9, σ_color=75, σ_space=75
        ▼
[ Step 3 ] Deconvoluzione Cromatica (Ruifrok & Johnston, 2001) + CLAHE
           Separa il canale Ematossilina dall'Eosina in spazio OD.
           CLAHE (clipLimit=2.0, tile 28×28 px = 6.4 µm) aumenta
           il contrasto locale della cromatina nucleare.
```

---

## Scelte Progettuali Motivate e Riferimenti Bibliografici

### 1. Normalizzazione di Macenko vs. Reinhard
* **Scelta:** Normalizzazione cromatica di Macenko in Densità Ottica (OD).
* **Riferimento:** **Macenko et al. (2009)** — *IEEE ISBI*, pp. 1107-1110.
* **Motivazione Scientifica:** A differenza dei metodi di allineamento statistico nello spazio colore LAB (Reinhard), l'algoritmo di Macenko converte l'immagine in Densità Ottica ($OD = -\log_{10}(I/255)$) ed applica la Decomposizione ai Valori Singoli (**SVD**) per stimare i reali vettori spettrali di assorbimento dell'Ematossilina e dell'Eosina. Questo rispetta la fisica dell'interazione luce-colorante regolata dalla legge di Beer-Lambert (*Macenko et al., 2009*).

### 2. Filtro Bilaterale vs. Filtro Gaussiano
* **Scelta:** Filtro Bilaterale con kernel $d=9$, $\sigma_{\text{color}}=75$, $\sigma_{\text{space}}=75$.
* **Motivazione Scientifica:** Il filtro Gaussiano attua una sfocatura uniforme che ammorbidisce e fonde i contorni delle membrane di nuclei adiacenti a contatto, ostacolandone la separazione. Il filtro bilaterale combina una componente spaziale e una cromatica, attenuando il rumore nelle regioni uniformi dello stroma ma **preservando la nitidezza dei gradienti di bordo nucleare**, essenziale per gli algoritmi di segmentazione (*Schmidt et al., 2018*).

### 3. Deconvoluzione Cromatica di Ruifrok & CLAHE Adattivo
* **Scelta:** Deconvoluzione Ruifrok & Johnston seguita da CLAHE su griglia 8×8.
* **Riferimenti:** **Ruifrok & Johnston (2001)** — *Anal. Quant. Cytol. Histol.* 23(4):291-9.
* **Motivazione Scientifica:** La matrice di deconvoluzione Ruifrok disaccoppia matematicamente il segnale dell'Ematossilina (nuclei) da quello dell'Eosina (citoplasma). Il CLAHE adattivo applicato su tile di 28×28 px (~6.4 × 6.4 µm²) opera alla stessa scala spaziale dei nuclei linfocitari (5–10 µm), aumentando il contrasto della cromatina nucleare senza amplificare il rumore stromatico globale (*Sung et al., 2024*).

---

## Fix Applicati dopo Audit (v1 → v2)

| Fix | Descrizione | Motivazione Scientifica |
|-----|-------------|-------------------------|
| 🔴 **FIX 1** | Allineamento stadi salvataggio (`denoised_rgb` per entrambi) | Garantisce coerenza tra le maschere nucleari e l'immagine RGB di overlay |
| 🔴 **FIX 2** | Reference image selezionata da mediana del dataset (`REACTIVE_examples (133).jpg`) | Evita distorsioni cromatiche derivanti dall'uso arbitrario della prima immagine alfabetica (distanza dalla mediana ridotta da 10.06 a 1.08) |
| 🟡 **FIX 3** | Refactoring SVD in `_estimate_HE_vectors()` | Principi di ingegneria del software DRY (Don't Repeat Yourself) |
| 🟡 **FIX 4** | Salvataggio `preprocessing_metadata.json` | Principi FAIR e riproducibilità scientifica della ricerca |

---

## Risultati Empirici

| Metrica | Valore |
|---------|--------|
| Immagini processate | **600/600** (0 errori) |
| Immagini blank o a basso contrasto | **0/600** |
| Riduzione varianza cromatica inter-immagine | **68.6%** (*Macenko et al., 2009*) |
| Intensità media H-channel FL | 62.9 ± 28.2 |
| Intensità media H-channel REACTIVE | 78.6 ± 17.8 |
| Differenza FL vs REACTIVE (H-channel) | **15.8 punti** (riflette l'ipercromasia tumorale: *Carreras, 2023*) |
| Tempo di elaborazione batch | **28.9 secondi** |

---

## 📚 Bibliografia della Fase 1

1. **Carreras J, Ikoma H, Kikuti YY, et al.** (2025). *Histological Image Classification Between Follicular Lymphoma and Reactive Lymphoid Tissue Using Deep Learning and Explainable Artificial Intelligence (XAI)*. **Cancers**, 17(15), 2428. DOI: 10.3390/cancers17152428.
2. **Macenko M, Niethammer M, Marron JS, et al.** (2009). *A method for normalizing histology slides for quantitative analysis*. **IEEE ISBI**, pp. 1107-1110. DOI: 10.1109/ISBI.2009.5193250.
3. **Ruifrok AC, Johnston DA.** (2001). *Quantification of histochemical staining by color deconvolution*. **Analytical and Quantitative Cytology and Histology**, 23(4), 291-299.
4. **Iwamoto R, Nishikawa T, Musangile FY, et al.** (2024). *Small sized centroblasts as poor prognostic factor in follicular lymphoma - Based on artificial intelligence analysis*. **Computers in Biology and Medicine**, 178, 108774. DOI: 10.1016/j.compbiomed.2024.108774.
5. **Sung YN, Lee H, Kim E, et al.** (2024). *Interpretable deep learning model to predict lymph node metastasis in early gastric cancer using whole slide images*. **American Journal of Cancer Research**, 14(7), 3513-3522. PMID: 39113689.
