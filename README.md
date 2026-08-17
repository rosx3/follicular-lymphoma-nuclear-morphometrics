# Quantificazione Citomorfometrica e Spaziale per la Classificazione Istologica tra Linfoma Follicolare e Tessuto Reattivo

> **Tesi di Laurea Triennale in Ingegneria / Scienza dei Dati**  
> **Approccio "White-Box" Interpretabile per la Patologia Digitale (Segmentazione $\rightarrow$ Biomarcatori $\rightarrow$ Machine Learning Tabulare $\rightarrow$ XAI)**

---

## 📌 Panoramica del Progetto

Il presente progetto sviluppa un sistema di supporto alla diagnosi istopatologica per la discriminazione differenziale tra **Linfoma Follicolare (FL)** e **Tessuto Linfoide Reattivo / Iperplasia Follicolare (REACTIVE)** a partire da immagini al microscopio colorate con Ematossilina ed Eosina (H&E).

A differenza dei classici approcci "Black-Box" basati su reti neurali convoluzionali end-to-end (*Carreras et al., 2025*), questo lavoro adotta un **paradigma White-Box guidato da biomarcatori fisici e spaziali**, strutturato in 4 fasi sequenziali:

```
[ Immagini H&E (600 Patch) ]
            │
            ▼
┌─────────────────────────┐
│  Fase 1: Preprocessing  │ ──► Normalizzazione Macenko + Filtro Bilaterale + CLAHE (H-channel)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Fase 2: Segmentazione  │ ──► Marker-Controlled Watershed (94.042 nuclei isolati) + U-Net
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Fase 3: Biomarcatori   │ ──► Morfometria, Grafi Spaziali (Delaunay/MST), Tessitura (GLCM/LBP)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Fase 4: Tabular ML & XAI│ ──► Random Forest / XGBoost + SHAP (Spiegabilità Clinica)
└─────────────────────────┘
```

---

## 📊 Dataset

Il dataset è composto da **600 patch istologiche H&E** alla risoluzione di $224 \times 224$ pixel, estratte da biopsie linfonodali (*Zenodo DOI: 10.5281/zenodo.15702609*):
- **300 patch** di **Linfoma Follicolare (FL)** (44.749 nuclei isolati)
- **300 patch** di **Tessuto Reattivo (REACTIVE)** (49.293 nuclei isolati)
- **Risoluzione Spaziale:** $0.23\,\mu m/\text{pixel}$ (calibrazione scanner $40\times$)

---

## 🗂️ Struttura della Repository

```text
testNuovoTesi/
├── data/                                  # Dati numerici e output di pipeline
│   ├── raw/                               # Dataset grezzo di input
│   ├── fase1_preprocessing/               # Immagini normalizzate Macenko e canali H
│   ├── ground_truth/                      # 30 patch di validazione benchmark
│   └── fase2_segmentation/                # Maschere d'istanza (16-bit PNG) e centroidi
│       ├── centroids_all.csv              # CSV Master: 94.042 nuclei con coord (x,y px/µm) e area
│       ├── colab_benchmark_results.csv    # Benchmark quantitativo vs Cellpose GT
│       └── segmentation_metadata.json     # Metadati completi della Fase 2
│
├── img/                                   # Grafici, anteprime e visualizzazioni visive
│   ├── fase1/                             # Preview normalizzazione e separazione croma
│   └── fase2/                             # Preview segmentazione, overlay contorni e loss
│
├── reports/                               # Report scientifici dettagliati in Markdown
│   ├── fase1_report.md                    # Auditing e risultati Preprocessing
│   └── fase2_report.md                    # Auditing, Metriche AJI/F1 e Benchmark Segmentazione
│
├── src/                                   # Codice sorgente modulare Python
│   ├── 01_preprocessing.py                # Pipeline Macenko + Deconvoluzione H-channel
│   ├── 02_segmentation.py                 # Marker-Controlled Watershed v3.0 + U-Net PyTorch
│   ├── 03_feature_extraction.py           # Estrazione biomarcatori morfometrici/spaziali
│   └── 04_classification.py               # Machine Learning Tabulare & XAI (SHAP)
│
├── Biblioteca personale.txt               # Riferimenti bibliografici accademici
└── README.md                              # Documentazione principale della repository
```

---

## 🔬 Metodologia e Risultati Salienti

### Fase 1: Preprocessing e Deconvoluzione Cromatica
- **Normalizzazione di Macenko** (*Macenko et al., 2009*): Decomposizione del vettore di assorbimento della macchia in spazio OD via SVD.
- **Isolamento del canale H (Ematossilina):** Estrazione del segnale nucleare purificato, trattato con Filtro Bilaterale (riduzione del rumore preservando i bordi) e **CLAHE** per l'esaltazione del contrasto locale.

### Fase 2: Segmentazione d'Istanza dei Nuclei Cellulari
- **Algoritmo Operativo:** Marker-Controlled Distance-Transform Watershed ($12\text{ px} \approx 2.8\,\mu m$ min distance, max area $2500\text{ px} \approx 132\,\mu m^2$).
- **Nuclei Estratti:** **94.042 nuclei totali** registrati nel file `centroids_all.csv`.
- **Validazione Indipendente su GPU (Cellpose v4.x Oracle GT, $d=22.0\text{ px} \approx 5.06\,\mu m$):**
  - **Dice Score (Pixel-level):** Watershed **$63.73\% \pm 10.91\%$** vs U-Net ResNet-34 **$57.38\% \pm 12.60\%$**
  - **AJI Index (Instance-level):** Watershed **$0.3255 \pm 0.0646$** vs U-Net ResNet-34 **$0.2873 \pm 0.0645$**
  - **F1 Detection Score:** Watershed **$0.4101 \pm 0.0716$** vs U-Net ResNet-34 **$0.3508 \pm 0.0882$**
- **Risultato Principale:** Il Watershed zero-shot guidato dalla fisica dell'assorbimento cromatico supera le prestazioni della U-Net deep learning, dimostrandosi immune all'overfitting da campioni limitati.

---

## 🛠️ Requisiti e Installazione

### Requisiti Ambiente Locale
- **Python:** $\ge 3.10$
- **Librerie Principali:** `opencv-python`, `scikit-image`, `scipy`, `numpy`, `torch`, `torchvision`, `matplotlib`

```bash
# Clone o navigazione nella cartella
cd testNuovoTesi

# Esecuzione Preprocessing (Fase 1)
python src/01_preprocessing.py

# Esecuzione Segmentazione Nuclei (Fase 2)
python src/02_segmentation.py
```

---

## 📚 Bibliografia Principale

1. **Carreras J, Ikoma H, Kikuti YY, et al.** (2025). *Histological Image Classification Between Follicular Lymphoma and Reactive Lymphoid Tissue Using Deep Learning and Explainable Artificial Intelligence (XAI)*. **Cancers**, 17(15), 2428. DOI: 10.3390/cancers17152428.
2. **Macenko M, Niethammer M, Marron JS, et al.** (2009). *A method for normalizing histology slides for quantitative analysis*. **IEEE ISBI 2009**, pp. 1107-1110. DOI: 10.1109/ISBI.2009.5193250.
3. **Kumar N, Verma R, Sharma S, et al.** (2017). *A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology*. **IEEE Transactions on Medical Imaging**, 36(7), 1550-1560. DOI: 10.1109/TMI.2017.2677499.
4. **Stringer C, Wang T, Michaelos M, Pachitariu M.** (2021). *Cellpose: a generalist algorithm for cellular segmentation*. **Nature Methods**, 18, 100-106. DOI: 10.1038/s41592-020-01018-x.
5. **Iwamoto R, Nishikawa T, Musangile FY, et al.** (2024). *Small sized centroblasts as poor prognostic factor in follicular lymphoma*. **Computers in Biology and Medicine**, 178, 108774. DOI: 10.1016/j.compbiomed.2024.108774.
6. **Schmidt U, Weigert M, Broaddus C, Myers G.** (2018). *Cell Detection with Star-convex Polygons*. **MICCAI 2018**, LNCS 11071, pp. 265-273.
