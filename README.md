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
│  Fase 3: Biomarcatori   │ ──► Morfometria, Micro-spazialità (k-NN), Tessitura (GLCM/LBP)
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
- **Risoluzione Spaziale:** $0.46\,\mu m/\text{pixel}$ — patch esportate a $200\times$ (obiettivo $20\times$) dallo scanner Hamamatsu NanoZoomer S360. Campo visivo $103.04 \times 103.04\,\mu m$. Il valore è **dedotto** dalle condizioni di esportazione dichiarate in *Carreras et al. (2025)*, che non pubblicano una scala esplicita: derivazione completa in [`reports/fase1_report.md`](reports/fase1_report.md#come-è-stata-determinata-la-scala-spaziale).

---

## 🗂️ Struttura della Repository

```text
testNuovoTesi/
├── data/                                  # Dati numerici e output di pipeline
│   ├── raw/                               # Dataset grezzo di input
│   ├── fase1_preprocessing/               # Immagini normalizzate Macenko e canali H
│   ├── ground_truth/                      # 30 patch di validazione benchmark
│   ├── fase2_segmentation/                # Maschere d'istanza (16-bit PNG) e centroidi
│   │   ├── centroids_all.csv              # CSV Master: 94.042 nuclei con coord (x,y px/µm) e area
│   │   ├── colab_benchmark_results.csv    # Benchmark quantitativo vs Cellpose GT
│   │   └── segmentation_metadata.json     # Metadati completi della Fase 2
│   └── fase3_features/                    # Matrice tabulare dei biomarcatori
│       ├── features_patches_master.csv    # ⭐ 600 patch × 50 colonne — input della Fase 4
│       ├── features_nuclei_all.csv        # 94.042 nuclei, morfometria per singolo nucleo
│       ├── separability_tests.csv         # Test FL vs REACTIVE con correzione FDR
│       └── feature_extraction_metadata.json  # Parametri e ambiente (riproducibilità)
│
├── img/                                   # Grafici, anteprime e visualizzazioni visive
│   ├── fase1/                             # Preview normalizzazione e separazione croma
│   ├── fase2/                             # Preview segmentazione, overlay contorni e loss
│   └── fase3/                             # Boxplot, heatmap correlazione, distribuzioni k-NN
│
├── reports/                               # Report scientifici dettagliati in Markdown
│   ├── fase1_report.md                    # Preprocessing e derivazione della scala spaziale
│   ├── fase2_report.md                    # Metriche AJI/F1 e Benchmark Segmentazione
│   ├── fase3_report.md                    # Biomarcatori, separabilità statistica e figure
│   └── fase3_implementation_plan.md       # Piano operativo, decisioni D1–D7, stato dei task
│
├── src/                                   # Codice sorgente modulare Python
│   ├── run_pipeline.py                    # ⭐ Entry point principale — esegue l'intera pipeline
│   ├── naming.py                          # Convenzioni di naming, categorie e risoluzione percorsi
│   ├── calibration.py                     # ⭐ Calibrazione spaziale — unica fonte di verità
│   ├── 01_preprocessing.py                # Pipeline Macenko + Deconvoluzione H-channel
│   ├── 02_segmentation.py                 # Marker-Controlled Watershed v3.0 + U-Net PyTorch
│   ├── 03_feature_extraction.py           # Estrazione biomarcatori morfometrici/spaziali
│   ├── feature_analysis.py                # Test di separabilità FL vs REACTIVE e figure
│   └── 04_classification.py               # Machine Learning Tabulare & XAI (SHAP)
│
├── tests/                                 # Test automatici (pytest) — 117 test
│   ├── test_calibration.py                # Calibrazione e assenza di duplicazioni
│   ├── test_naming.py                     # Convenzioni di naming e categorie
│   ├── test_pipeline_paths.py             # Risoluzione input delle 600 patch
│   ├── test_pipeline_end_to_end.py        # Catena Fase 1 → 2 → 3 su mini-dataset
│   ├── test_feature_knn.py                # Distanze micro-spaziali
│   ├── test_feature_texture.py            # GLCM/LBP mascherati sui nuclei
│   ├── test_patch_feature_contract.py     # Contratto delle 50 colonne
│   ├── test_feature_analysis.py           # Test statistici e correzione FDR
│   ├── test_feature_figures.py            # Figure per la tesi
│   └── test_segmentation_split.py         # Split stratificato train/val
│
├── Biblioteca personale.txt               # Riferimenti bibliografici accademici
├── requirements.txt                       # Dipendenze Python con versioni pinned
└── README.md                              # Documentazione principale della repository
```

---

## 🔬 Metodologia e Risultati Salienti

### Fase 1: Preprocessing e Deconvoluzione Cromatica
- **Normalizzazione di Macenko** (*Macenko et al., 2009*): Decomposizione del vettore di assorbimento della macchia in spazio OD via SVD.
- **Isolamento del canale H (Ematossilina):** Estrazione del segnale nucleare purificato, trattato con Filtro Bilaterale (riduzione del rumore preservando i bordi) e **CLAHE** per l'esaltazione del contrasto locale.

### Fase 2: Segmentazione d'Istanza dei Nuclei Cellulari
- **Algoritmo Operativo:** Marker-Controlled Distance-Transform Watershed ($12\text{ px} \approx 5.5\,\mu m$ min distance, max area $2500\text{ px} \approx 529\,\mu m^2$).
- **Nuclei Estratti:** **94.042 nuclei totali** registrati nel file `centroids_all.csv`.
- **Validazione Indipendente su GPU (Cellpose v4.x Oracle GT, $d=22.0\text{ px} \approx 10.1\,\mu m$):**
  - **Dice Score (Pixel-level):** Watershed **$63.73\% \pm 10.91\%$** vs U-Net ResNet-34 **$57.38\% \pm 12.60\%$**
  - **AJI Index (Instance-level):** Watershed **$0.3097 \pm 0.0723$** vs U-Net ResNet-34 **$0.2873 \pm 0.0645$**
  - **F1 Detection Score:** Watershed **$0.4101 \pm 0.0716$** vs U-Net ResNet-34 **$0.3508 \pm 0.0882$**
- **Risultato Principale:** Il Watershed zero-shot guidato dalla fisica dell'assorbimento cromatico supera le prestazioni della U-Net deep learning, dimostrandosi immune all'overfitting da campioni limitati.

### Fase 3: Estrazione dei Biomarcatori e Separabilità Statistica
- **Matrice prodotta:** **600 patch × 50 colonne** (47 biomarcatori + 3 metadati), da 94.042 nuclei. Zero errori, zero valori mancanti.
- **Tre famiglie di biomarcatori:** citomorfometria aggregata (32 colonne: `mean`/`std`/`skew`/`cv` su 8 grandezze di forma e dimensione), densità nucleare (3), indicatori di Iwamoto et al. 2024 sul top 10% per area (2), distanze micro-spaziali k-NN (4) e tessitura cromatinica GLCM/LBP ristretta ai pixel nucleari (6).
- **Separabilità FL vs REACTIVE:** **37 biomarcatori su 47 significativi** (Mann-Whitney U o t-test di Welch scelti sulla normalità osservata, correzione FDR di Benjamini-Hochberg).
- **Biomarcatore più discriminante:** `lbp_entropy` con p (FDR) = 3.2e-51 ed effect size rango-biseriale −0.72 — la **complessità della micro-tessitura cromatinica** separa le due classi meglio di qualunque descrittore di forma o dimensione.
- **Quadro clinico coerente su tre fronti indipendenti:** nel linfoma follicolare i nuclei sono più piccoli e allungati, il packing è meno fitto (distanze inter-nucleari maggiori, densità minore) e la cromatina è più uniforme.
- **Non discriminanti:** l'intera famiglia della solidità e della circolarità — a questa scala le due popolazioni nucleari sono ugualmente compatte.
- **Nota metodologica:** durante questa fase è emersa ed è stata corretta un'assunzione errata sulla **calibrazione spaziale** (da 0.23 a 0.46 µm/px); la derivazione del valore corretto è documentata in [`reports/fase1_report.md`](reports/fase1_report.md).

---

## 🛠️ Requisiti e Installazione

### Requisiti Ambiente Locale
- **Ambiente di riferimento:** Python **3.14.3** su Windows 11, esecuzione su CPU (verificato il 19 agosto 2026). Il benchmark di segmentazione con Cellpose è stato eseguito su Google Colab / GPU Tesla T4.
- **Librerie Principali:** `numpy 2.4.3`, `scipy 1.17.1`, `scikit-image 0.26.0`, `opencv-python 5.0.0.93`, `torch 2.13.0`, `torchvision 0.28.0`, `matplotlib 3.10.8`
- Le versioni sono **fissate esattamente** in `requirements.txt`: sono quelle sotto cui sono stati prodotti i risultati numerici riportati nei report.

```bash
# 1. Installare le dipendenze (prima esecuzione)
pip install -r requirements.txt

# 2. Eseguire l'intera pipeline (Fasi 1, 2 e 3)
python src/run_pipeline.py

# 3. Analisi di separabilità statistica e figure della Fase 3
python src/feature_analysis.py

# Oppure, per eseguire solo una fase specifica:
python src/run_pipeline.py --fase 1        # Solo Preprocessing
python src/run_pipeline.py --fase 2        # Solo Segmentazione

# I singoli moduli possono essere eseguiti per i self-test interni:
python src/01_preprocessing.py            # [TEST] self-test modulo Preprocessing
python src/02_segmentation.py             # [TEST] self-test modulo Segmentazione

# Suite di test automatici
python -m pytest tests/ -q
```

### Convenzioni dei nomi di file

Tutti i nomi dei file intermedi e le etichette di categoria sono definiti in un
unico punto, `src/naming.py`, e verificati dai test contro i file realmente
presenti su disco:

| Fase | File prodotto | Categoria canonica |
|---|---|---|
| Fase 1 | `<stem>_norm.png`, `<stem>_hchannel.png` | `follicular_lymphoma` |
| Fase 2 | `<stem>_mask.png`, `<stem>_overlay.png` | `reactive_tissue` |

---

## 📚 Bibliografia Principale

1. **Carreras J, Ikoma H, Kikuti YY, et al.** (2025). *Histological Image Classification Between Follicular Lymphoma and Reactive Lymphoid Tissue Using Deep Learning and Explainable Artificial Intelligence (XAI)*. **Cancers**, 17(15), 2428. DOI: 10.3390/cancers17152428.
2. **Macenko M, Niethammer M, Marron JS, et al.** (2009). *A method for normalizing histology slides for quantitative analysis*. **IEEE ISBI 2009**, pp. 1107-1110. DOI: 10.1109/ISBI.2009.5193250.
3. **Kumar N, Verma R, Sharma S, et al.** (2017). *A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology*. **IEEE Transactions on Medical Imaging**, 36(7), 1550-1560. DOI: 10.1109/TMI.2017.2677499.
4. **Stringer C, Wang T, Michaelos M, Pachitariu M.** (2021). *Cellpose: a generalist algorithm for cellular segmentation*. **Nature Methods**, 18, 100-106. DOI: 10.1038/s41592-020-01018-x.
5. **Iwamoto R, Nishikawa T, Musangile FY, et al.** (2024). *Small sized centroblasts as poor prognostic factor in follicular lymphoma*. **Computers in Biology and Medicine**, 178, 108774. DOI: 10.1016/j.compbiomed.2024.108774.
6. **Schmidt U, Weigert M, Broaddus C, Myers G.** (2018). *Cell Detection with Star-convex Polygons*. **MICCAI 2018**, LNCS 11071, pp. 265-273.
