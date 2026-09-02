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
│  Fase 2: Segmentazione  │ ──► Marker-Controlled Watershed (94.042 nuclei isolati)
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
│   ├── ground_truth/                      # 30 patch di validazione benchmark + maschere Cellpose
│   ├── annotazione_manuale/               # ⭐ 1656 nuclei marcati a mano, protocollo e accordo con gli algoritmi
│   ├── fase2_segmentation/                # Maschere d'istanza (16-bit PNG) e centroidi
│   │   ├── centroids_all.csv              # CSV Master: 94.042 nuclei con coord (x,y px/µm) e area
│   │   ├── colab_benchmark_results.csv    # Benchmark quantitativo vs Cellpose GT
│   │   └── segmentation_metadata.json     # Metadati completi della Fase 2
│   ├── fase3_features/                    # Matrice tabulare dei biomarcatori
│   │   ├── features_patches_master.csv    # ⭐ 600 patch × 50 colonne — input della Fase 4
│   │   ├── features_nuclei_all.csv        # 94.042 nuclei, morfometria per singolo nucleo
│   │   ├── separability_tests.csv         # Test FL vs REACTIVE con correzione FDR
│   │   └── feature_extraction_metadata.json  # Parametri e ambiente (riproducibilità)
│   └── fase4_classification/              # Metriche, modello, SHAP, contributo per famiglia e robustezza
│
├── img/                                   # Grafici, anteprime e visualizzazioni visive
│   ├── fase1/                             # Preview normalizzazione e separazione croma
│   ├── fase2/                             # Preview segmentazione, overlay contorni e loss
│   ├── fase3/                             # Boxplot, heatmap correlazione, distribuzioni k-NN
│   └── fase4/                             # ROC, forbice, SHAP, robustezza alla colorazione
│
├── reports/                               # Report scientifici dettagliati in Markdown
│   ├── fase1_report.md                    # Preprocessing e derivazione della scala spaziale
│   ├── fase2_report.md                    # Metriche AJI/F1 e Benchmark Segmentazione
│   ├── fase3_report.md                    # Biomarcatori, separabilità statistica e figure
│   ├── fase4_report.md                    # Classificazione, forbice di validazione e SHAP
│   └── fase3_implementation_plan.md       # Piano operativo, decisioni D1–D7, stato dei task
│
├── src/                                   # Codice sorgente modulare Python
│   ├── run_pipeline.py                    # ⭐ Entry point principale — esegue l'intera pipeline
│   ├── naming.py                          # Convenzioni di naming, categorie e risoluzione percorsi
│   ├── calibration.py                     # ⭐ Calibrazione spaziale — unica fonte di verità
│   ├── 01_preprocessing.py                # Pipeline Macenko + Deconvoluzione H-channel
│   ├── 02_segmentation.py                 # Marker-Controlled Watershed v4.3
│   ├── 03_feature_extraction.py           # Estrazione biomarcatori morfometrici/spaziali
│   ├── feature_analysis.py                # Test di separabilità FL vs REACTIVE e figure
│   ├── 04_classification.py               # ⭐ ML Tabulare & XAI — doppia validazione, SHAP, contributo per famiglia
│   ├── stain_robustness.py                # La tessitura legge la cromatina o il vetrino?
│   ├── block_structure.py                 # L'ordine di numerazione conserva struttura? (premessa della validazione a blocchi)
│   ├── prepara_annotazione.py             # Esporta le patch per l'annotazione manuale
│   ├── annotation_agreement.py            # ⭐ Verifica umana del riferimento Cellpose (1656 nuclei)
│   ├── gui_core.py                        # Logica dell'interfaccia (riusa la pipeline, no duplicati)
│   └── gui.py                             # ⭐ Interfaccia Streamlit — `streamlit run src/gui.py`
│
├── tests/                                 # Test automatici (pytest) — 220 test
│   ├── test_calibration.py                # Calibrazione e assenza di duplicazioni
│   ├── test_naming.py                     # Convenzioni di naming e categorie
│   ├── test_pipeline_paths.py             # Risoluzione input delle 600 patch
│   ├── test_pipeline_end_to_end.py        # Catena Fase 1 → 2 → 3 su mini-dataset
│   ├── test_feature_knn.py                # Distanze micro-spaziali
│   ├── test_feature_texture.py            # GLCM/LBP mascherati sui nuclei
│   ├── test_patch_feature_contract.py     # Contratto delle 50 colonne
│   ├── test_feature_analysis.py           # Test statistici e correzione FDR
│   ├── test_feature_figures.py            # Figure per la tesi
│   ├── test_block_structure.py            # L'ordine di numerazione conserva struttura?
│   ├── test_gui_core.py                   # Coerenza GUI ↔ pipeline sui biomarcatori
│   ├── test_gui.py                        # Interfaccia Streamlit headless (AppTest)
│   ├── test_gui_e2e.py                    # Interfaccia con browser vero (Playwright)
│   ├── test_classification.py             # Fase 4: blocchi, riduzione ridondanze, SHAP
│   ├── test_stain_robustness.py           # La perturbazione altera il colore, non la geometria
│   └── test_segmentation_reproducibility.py  # I default rigenerano le maschere del dataset
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
- **Algoritmo Operativo:** Marker-Controlled Distance-Transform Watershed ($7\text{ px} \approx 3.2\,\mu m$ min distance, area nucleare ammessa $15$–$2500\text{ px}$).
- **Riproducibilità dei parametri (agosto 2026):** le 600 maschere furono generate da un runner esterno al repository, prima che `run_pipeline.py` esistesse, e i suoi parametri non erano registrati da nessuna parte. I default rimasti nel modulo non erano quelli usati, e rieseguire la Fase 2 produceva il **54% di nuclei in meno**. I valori originali sono stati ricostruiti per ricerca su griglia e verificati su 60 patch estratte a caso: **60/60 identiche pixel per pixel** alle maschere del dataset. `tests/test_segmentation_reproducibility.py` impedisce che si perdano di nuovo.
- **Nuclei Estratti:** **94.042 nuclei totali** registrati nel file `centroids_all.csv`.
- **Validazione Indipendente su GPU** (Cellpose Oracle GT, $d=22.0\text{ px} \approx 10.1\,\mu m$; $n=10$ patch di validazione, run del 20 agosto 2026):

  | Metrica | Accordo Watershed vs Cellpose |
  |---|:---:|
  | Dice (pixel) | $0.7950 \pm 0.0442$ |
  | IoU (pixel) | $0.6620 \pm 0.0598$ |
  | AJI (istanza) | $0.5411 \pm 0.0730$ |
  | F1 detection @0.5 | $0.7108 \pm 0.0718$ |

- **Verifica umana del riferimento** (2 settembre 2026): 1656 nuclei marcati a mano, alla cieca, su tutte e 10 le patch di validazione. Cellpose copre il **95,2%** dei nuclei riconosciuti dal lettore, il Watershed l'**87,9%**. Il limite del Watershed non è la rilevazione ma la **fusione dei nuclei addossati**: 83 casi, presenti in 10 immagini su 10, che coinvolgono il 10,3% dei nuclei marcati. Cellpose ne fonde 2 in totale. Confronti appaiati $p = 0.002$, il minimo ottenibile con $n=10$. Dettagli in [`reports/fase2_report.md`](reports/fase2_report.md) §3.3.

- **Risultato Principale:** il conteggio complessivo del Watershed coincide con quello umano (rapporto mediano $1.001$), ma **l'accordo è apparente**: nasce da due errori di verso opposto che si compensano, nuclei fusi da un lato e oggetti aggiunti dall'altro. Un confronto basato sui soli conteggi avrebbe concluso per un accordo perfetto.

> **Nota storica — perché i numeri sono cambiati.** Fino al 20 agosto 2026 questa sezione riportava Dice $0.6373$ vs $0.5738$ e concludeva per la superiorità del Watershed. Quel benchmark invocava la segmentazione senza parametri espliciti e misurava quindi `min_distance=12, min_area_px=30`, **non** i parametri con cui è stato costruito il dataset (verificato sui conteggi `ws_n_pred`: 10 patch su 10). Corretti i parametri, il Watershed migliora nettamente, +66% di AJI e +73% di F1. Analisi completa in [`reports/fase2_report.md`](reports/fase2_report.md) §7.8.

### Fase 3: Estrazione dei Biomarcatori e Separabilità Statistica
- **Matrice prodotta:** **600 patch × 50 colonne** (47 biomarcatori + 3 metadati), da 94.042 nuclei. Zero errori, zero valori mancanti.
- **Tre famiglie di biomarcatori:** citomorfometria aggregata (32 colonne: `mean`/`std`/`skew`/`cv` su 8 grandezze di forma e dimensione), densità nucleare (3), indicatori di Iwamoto et al. 2024 sul top 10% per area (2), distanze micro-spaziali k-NN (4) e tessitura cromatinica GLCM/LBP ristretta ai pixel nucleari (6).
- **Separabilità FL vs REACTIVE:** **37 biomarcatori su 47 significativi** (Mann-Whitney U o t-test di Welch scelti sulla normalità osservata, correzione FDR di Benjamini-Hochberg).
- **Biomarcatore più discriminante:** `lbp_entropy` con p (FDR) = 3.2e-51 ed effect size rango-biseriale −0.72 — la **complessità della micro-tessitura cromatinica** separa le due classi meglio di qualunque descrittore di forma o dimensione.
- **Quadro clinico coerente su tre fronti indipendenti:** nel linfoma follicolare i nuclei sono più piccoli e allungati, il packing è meno fitto (distanze inter-nucleari maggiori, densità minore) e la cromatina è più uniforme.
- **Non discriminanti:** l'intera famiglia della solidità e della circolarità — a questa scala le due popolazioni nucleari sono ugualmente compatte.
- **Nota metodologica:** durante questa fase è emersa ed è stata corretta un'assunzione errata sulla **calibrazione spaziale** (da 0.23 a 0.46 µm/px); la derivazione del valore corretto è documentata in [`reports/fase1_report.md`](reports/fase1_report.md).

### Fase 4: Classificazione Tabulare e Spiegabilità Clinica

- **Biomarcatori usati:** **33 dei 47**, ottenuti raggruppando le variabili quasi identiche ($|\rho| > 0.90$) e tenendo di ogni gruppo la più leggibile clinicamente. La riduzione serve alla spiegabilità: fra due variabili ridondanti SHAP divide il merito arbitrariamente e le fa apparire entrambe meno importanti di quanto sono.
- **Doppia validazione.** Le 600 patch provengono da ~221 casi, con più patch per caso, ma il dataset pubblicato non contiene identificativi di paziente. Ogni modello è quindi valutato due volte: con split casuale (ottimistico) e con split a blocchi contigui che tengono unite le patch vicine (conservativo). **Quel che si pubblica è la forbice fra i due.**

  | Modello | Split casuale | **Blocchi (conservativo)** | Forbice |
  |---|:---:|:---:|:---:|
  | Regressione logistica | $0.9181$ | $0.8992$ | $+0.019$ |
  | Random Forest | $0.9602$ | $0.9361$ | $+0.024$ |
  | **XGBoost** | $0.9648$ | $\mathbf{0.9401\ [0.9057,\ 0.9744]}$ | $+0.025$ |

- **Il leakage valeva circa due punti di AUC**, misurati anziché stimati a occhio. Il degrado è monotono al crescere del blocco (XGBoost: $0.960 \rightarrow 0.935$ da blocchi di 5 a 30), il che conferma che la dipendenza dal vicinato è reale.
- **Complessità e interpretabilità:** Random Forest e XGBoost sono statisticamente indistinguibili ($p = 1.000$), e una regressione logistica arriva a $0.899$. Gran parte del segnale è lineare: il valore della fase non sta nel punteggio ma nel dire *quali* biomarcatori decidono.
- **Biomarcatore dominante:** `lbp_entropy` (importanza SHAP $3.15$, il doppio del secondo), primo anche nei test univariati della Fase 3.
- **Scoperta multivariata:** `solidity_mean` è terza per SHAP ma trentanovesima in Fase 3. Le medie di classe sono quasi identiche ($p = 0.106$) ma le **dispersioni** no (Levene $p = 3.0 \times 10^{-6}$): nel linfoma la solidità nucleare è più eterogenea, e valori estremi in entrambe le direzioni indicano FL. Un effetto di dispersione che il confronto fra medie non poteva rilevare — il **pleomorfismo nucleare**.
- **Gli errori non sono sparsi:** 28 blocchi su 60 non sbagliano nulla, 4 sbagliano più della metà. Il modello fallisce su pochi casi difficili, e su quelli fallisce quasi sempre.
- **Contributo per famiglia — a decidere è la tessitura, non la morfometria.** Cinque biomarcatori di tessitura e intensità arrivano da soli a $0.944$; i 28 morfometrici e spaziali si fermano a $0.857$. Indicazione coerente su tre modelli, non un confronto statisticamente dimostrato (report §4.4).

  | Sottoinsieme | n | XGBoost |
  |---|:---:|:---:|
  | Tutte | 33 | $0.940$ |
  | Senza intensità (`hchannel_*`) | 31 | $0.923$ |
  | Senza tessitura (GLCM, LBP) | 30 | $0.869$ |
  | Solo morfometria e spaziale | 28 | $0.857$ |
  | **Solo tessitura e intensità** | **5** | $\mathbf{0.944}$ |

  Il peso sta nel **pattern della cromatina** (GLCM, LBP), non nell'intensità della colorazione: togliere `hchannel_*` costa $0.017$, togliere GLCM e LBP quattro volte tanto.

- **Robustezza alla colorazione — legge la cromatina, non il vetrino.** Perturbando artificialmente la colorazione delle immagini grezze (Tellez et al., 2019) e rifacendo girare l'intera pipeline, a $\sigma = 0.2$ il modello perde mezzo punto di AUC e cambia classe su 2 patch su 100; a $\sigma = 0.3$ perde $1{,}6$ punti e il $92\%$ delle patch tiene. È la verifica che il risultato non poggia sul lotto di colorazione, e una giustificazione sperimentale a posteriori della normalizzazione di Macenko della Fase 1.

---

## 🛠️ Requisiti e Installazione

### Requisiti Ambiente Locale
- **Ambiente di riferimento:** Python **3.14.3** su Windows 11, esecuzione su CPU (verificato il 19 agosto 2026). Il benchmark di segmentazione con Cellpose è stato eseguito su Google Colab / GPU Tesla T4.
- **Librerie Principali:** `numpy 2.4.3`, `scipy 1.17.1`, `scikit-image 0.26.0`, `opencv-python 5.0.0.93`, `torch 2.13.0`, `torchvision 0.28.0`, `matplotlib 3.10.8`
- Le versioni sono **fissate esattamente** in `requirements.txt`: sono quelle sotto cui sono stati prodotti i risultati numerici riportati nei report.

> **Su Windows, se `python` non funziona.** Windows installa un alias verso il
> Microsoft Store che intercetta il comando e risponde *«Python non è stato
> trovato»* anche quando Python è regolarmente installato. In quel caso **usare
> `py` al posto di `python`** in tutti i comandi di questo README:
> `py -m pytest tests/ -q`, `py src/run_pipeline.py`, `py -m streamlit run src/gui.py`.
> In alternativa si disattiva l'alias da *Impostazioni → App → Impostazioni app
> avanzate → Alias di esecuzione dell'app*.

```bash
# 1. Installare le dipendenze (prima esecuzione)
pip install -r requirements.txt

# 2. Eseguire l'intera pipeline (Fasi 1, 2 e 3)
python src/run_pipeline.py

# 3. Analisi di separabilità statistica e figure della Fase 3
python src/feature_analysis.py

# 4. Classificazione tabulare e spiegabilità SHAP (Fase 4)
python src/04_classification.py

# Oppure, per eseguire solo una fase specifica:
python src/run_pipeline.py --fase 1        # Solo Preprocessing
python src/run_pipeline.py --fase 2        # Solo Segmentazione

# I singoli moduli possono essere eseguiti per i self-test interni:
python src/01_preprocessing.py            # [TEST] self-test modulo Preprocessing
python src/02_segmentation.py             # [TEST] self-test modulo Segmentazione

# Suite di test automatici (rapida: esclude i test col browser)
python -m pytest tests/ -q -m "not e2e"
```

#### Test end-to-end dell'interfaccia

La suite rapida collauda la GUI con `streamlit.testing.v1.AppTest`, che esegue
l'app senza browser: veloce, ma **non sa simulare il caricamento di un file**,
lasciando scoperto il percorso principale dell'interfaccia. Quel percorso è
coperto da test separati che aprono davvero un browser e caricano davvero
un'immagine dal disco.

```bash
pip install pytest-playwright && python -m playwright install chromium
python -m pytest tests/test_gui_e2e.py -q
```

### Interfaccia grafica

```bash
streamlit run src/gui.py
```

L'interfaccia richiede che le Fasi 1–3 siano già state eseguite (legge i loro
output da `data/`) e offre tre sezioni:

| Sezione | Cosa mostra |
|---|---|
| **Esplora dataset** | I cinque stadi della pipeline su una patch a scelta e i 47 biomarcatori, ciascuno posizionato nella distribuzione della propria classe |
| **Analizza immagine** | Un'immagine nuova percorre Fase 1 → 2 → 3 dal vivo, viene classificata e la decisione viene spiegata |
| **Spiegabilità** | La forbice fra le due validazioni, quali biomarcatori decidono e in che direzione, e la spiegazione locale di un caso a scelta calcolata dal vivo |
| **Risultati Fase 3** | Test di separabilità con correzione FDR e le figure prodotte da `feature_analysis.py` |

> **Non è un dispositivo diagnostico.** Il modello è addestrato su 600 patch di
> due sole classi: è uno strumento di ricerca. Su un'immagine che non sia una
> patch linfonodale H&E risponde comunque, con una sicurezza priva di fondamento.
> Per le patch del dataset l'interfaccia mostra la predizione **fuori-piega**, la
> sola onesta per un'immagine che il modello finale ha visto in addestramento.

La logica sta in `src/gui_core.py`, separata dai widget: chiama le stesse
funzioni di `run_pipeline.py` invece di reimplementarle, e un test di coerenza
verifica che elaborando dalla GUI una patch del dataset si riottengano i valori
scritti in `features_patches_master.csv`.

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
