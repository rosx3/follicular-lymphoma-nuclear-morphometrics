# Report Fase 3 — Estrazione Biomarcatori Citomorfometrici, Spaziali e di Tessitura
### Tesi: Quantificazione Citomorfometrica e Spaziale per la Classificazione tra Linfoma Follicolare e Tessuto Reattivo
*Modulo: src/03_feature_extraction.py*
*Aggiornato: 19 agosto 2026 — Corretta discrepanza nel conteggio delle feature (vedi nota in Sezione 2)*

---

## Obiettivo

Tradurre le **94.042 maschere d-istanza nucleare** prodotte dalla Fase 2 e i **600 H-channel CLAHE** prodotti dalla Fase 1 in una matrice tabulare di **biomarcatori fisici e clinicamente interpretabili**, espressi in unita reali (um, um2) con calibrazione 1 px = 0.23 um (scanner Hamamatsu NanoZoomer S360, obiettivo 40x).

Output:
- data/fase3_features/features_nuclei_all.csv
- data/fase3_features/features_patches_master.csv — 600 righe x 50 colonne (47 feature + 3 metadati)
- data/fase3_features/feature_extraction_metadata.json
- img/fase3/morphometry_regions_preview.png

---

## 1. Analisi Critica delle Feature: Cosa e Stato Incluso e Cosa Rimosso

### 1.1 Feature Rimosse e Motivazione

| Feature Rimossa | Motivo della Rimozione | Fonte della Decisione |
|---|---|---|
| equivalent_diameter_um | Ridondante matematicamente: e una trasformazione diretta di area_um2 (d = 2*sqrt(A/pi)). | Principio matematico |
| extent | Ridondante con solidity: entrambe misurano compattezza. Non citata in letteratura FL-specifica. | Analisi bibliografica |
| _median (tutte le feature) | Con 600 patch, mean+std+skew+cv copre gia il profilo statistico. Mediana e quasi equivalente alla media. | Analisi statistica |
| knn5_dist_mean/std_um | Fortemente correlato con k=3 alla scala micro-locale di 51.5 um2. | Analisi scala spaziale dataset |
| Triangolazione di Delaunay (5 feature) | Boundary effects critici: il grafo viene troncato ai bordi della patch (224x224 px = 51.5 um). Delaunay/MST hanno validita su WSI (ordine mm), non su micro-patch. | Letteratura 2024: edge effects at patch boundary |
| Minimum Spanning Tree (4 feature) | Stessa motivazione Delaunay. mst_edge_length_mean e quasi equivalente a knn1_dist_mean su distribuzioni quasi-uniformi. | Letteratura 2024 + analisi matematica |
| glcm_correlation | Alta correlazione con glcm_homogeneity su immagini H&E normalizzate. | Haralick (1973) |
| glcm_dissimilarity | Fortemente correlata con glcm_contrast. | Haralick (1973) |
| lbp_mean, lbp_std | Solo l-entropia LBP e robusta e interpretabile come misura di complessita della micro-tessitura. | Ojala et al. (2002) |
| hchannel_skew | Ridondante con area_um2_skew e circularity_skew. Dopo normalizzazione Macenko la varianza cromatica e gia ridotta del 68.6%. | reports/fase1_report.md |
| Momenti Cromatici CIE-LAB (3 feature) | Completamente ridondanti post-normalizzazione Macenko: lo scopo di Macenko e portare tutte le patch nello stesso spazio cromatico. Misurare i colori dopo = misurare il rumore residuo. | reports/fase1_report.md + letteratura |

NOTA PER LA TESI (Sezione Discussione):
Le feature Delaunay e MST sono scientificamente valide per l-analisi della micro-architettura tissutale,
ma richiedono un campo visivo nell-ordine dei mm (Whole Slide Images). Su patch 224x224 px (51.5 um)
i boundary effects ne compromettono l-affidabilita. Questa e una direzione futura per estendere l-analisi a WSI completi.

---

## 2. Set Definitivo dei Biomarcatori (47 feature + 3 metadati)

> **NOTA DI CORREZIONE (19 agosto 2026):** la versione precedente di questa sezione dichiarava
> erroneamente "51 feature". La tabella dettagliata di §2.4 elencava conteggi disomogenei per
> feature (somma reale: 19 colonne, non 32), mentre il testo introduttivo della stessa sezione
> dichiarava l'applicazione uniforme di 4 statistiche (mean/std/skew/cv) a tutte le 8 feature di
> base (8×4 = 32 colonne). Anche usando 32 (l'interpretazione effettivamente implementata in
> `src/03_feature_extraction.py` e verificata con self-test), la somma dei sub-totali di sezione
> (3 + 2 + 32 + 4 + 6) da **47**, non 51. Si è deciso di correggere il totale dichiarato a 47
> anziché aggiungere feature non pianificate, per mantenere il set minimale e motivato descritto
> in Sezione 1. La tabella di §2.4 sotto è stata aggiornata per riflettere il conteggio uniforme
> (4 colonne per ciascuna delle 8 feature di base) realmente implementato nel codice.

### 2.1 Metadati (3 colonne)
| Colonna | Tipo | Descrizione |
|---|---|---|
| image_name | str | Nome della patch sorgente |
| category | str | follicular_lymphoma / reactive_tissue |
| target | int | 1 = FL, 0 = REACTIVE |

### 2.2 Densita Nucleare (3 colonne)
| Feature | Unita | Descrizione |
|---|---|---|
| n_nuclei | count | Numero nuclei segmentati nella patch |
| nuclear_density_per_1000um2 | nuclei/1000um2 | Densita assoluta calibrata |
| nuclear_area_fraction | [0,1] | % area patch occupata da nuclei |

### 2.3 Biomarcatori Iwamoto et al. (2024) - Top 10% (2 colonne)
| Feature | Unita | p-value | Fonte |
|---|---|---|---|
| area_top10_mean_um2 | um2 | p=0.024 | Iwamoto et al. (2024), Computers in Biology and Medicine, Vol.178 |
| area_top10_short_axis_um | um | p=0.020 | Iwamoto et al. (2024), Computers in Biology and Medicine, Vol.178 |

### 2.4 Morfometria Aggregata per Patch (32 colonne)
Aggregati: _mean, _std, _skew, _cv per 8 feature di base:

| Feature di Base | Unita | N col (mean/std/skew/cv) | Fonte Clinica |
|---|---|---|---|
| area_um2 | um2 | 4 | Iwamoto et al. (2024), p=0.013 |
| perimeter_um | um | 4 | Standard morfometria |
| circularity | [0,1] | 4 | Centrociti = bassa circolarita |
| eccentricity | [0,1] | 4 | Allungamento centrociti |
| solidity | [0,1] | 4 | Concavita nucleari |
| major_axis_um | um | 4 | Iwamoto et al. (2024) long length, p=0.042 |
| minor_axis_um | um | 4 | Iwamoto et al. (2024) short length, p=0.007 |
| aspect_ratio | adim. | 4 | Rapporto forma major/minor |

### 2.5 Distanze k-NN (4 colonne)
| Feature | Unita | Descrizione |
|---|---|---|
| knn1_dist_mean_um | um | Distanza media al nucleo singolo piu vicino per tutti i nuclei della patch |
| knn1_dist_std_um | um | Deviazione standard - misura la regolarita del packing |
| knn3_dist_mean_um | um | Distanza media ai 3 nuclei piu vicini |
| knn3_dist_std_um | um | Deviazione standard vicinato k=3 |

### 2.6 Tessitura Cromatinica H-channel (6 colonne)
| Feature | Descrizione | Fonte |
|---|---|---|
| glcm_contrast | Variazione intensita cromatina (dispersa=alta) | Haralick et al. (1973) |
| glcm_homogeneity | Uniformita cromatina (compatta=alta) | Haralick et al. (1973) |
| glcm_energy | Ordine/uniformita (ASM) | Haralick et al. (1973) |
| lbp_entropy | Complessita micro-tessitura cromatina | Ojala et al. (2002) |
| hchannel_mean | Intensita media ematossilina nella patch | Standard patologia digitale |
| hchannel_std | Variabilita cromatina intra-patch | Standard patologia digitale |

---

## 3. Risultati dell-Estrazione - Statistiche Descrittive

[Sezione da completare dopo esecuzione di: python src/run_pipeline.py --fase 3]

### 3.1 Overview del Dataset Estratto
| Metrica | Valore |
|---|---|
| Patch processate | — / 600 |
| Nuclei elaborati | — / 94.042 |
| Feature estratte per patch | 47 (+ 3 metadati) |
| Valori NaN | — |
| Errori di processamento | — |
| Tempo di esecuzione | — s |

### 3.2 Statistiche Descrittive FL vs REACTIVE
[Tabella da generare dopo esecuzione]

### 3.3 Test Statistici di Separabilita (t-test / Mann-Whitney U)
[Tabella con p-value per ogni feature da generare dopo esecuzione]

---

## 4. Anteprime Grafiche

[Immagini da inserire dopo esecuzione]

---

## 5. Dipendenze e Riproducibilita

| Libreria | Uso |
|---|---|
| scipy.spatial.KDTree | k-NN distances |
| skimage.feature.graycomatrix/graycoprops | GLCM |
| skimage.feature.local_binary_pattern | LBP |
| skimage.measure.regionprops | Morfometria istanza |
| numpy | Aggregazioni statistiche |

---

## 6. Bibliografia

1. Iwamoto R, Nishikawa T, et al. (2024). Small sized centroblasts as poor prognostic factor in follicular lymphoma. Computers in Biology and Medicine, 178, 108774.
2. Haralick RM, Shanmugam K, Dinstein I. (1973). Textural features for image classification. IEEE Trans Systems, Man, Cybernetics, 3(6), 610-621.
3. Ojala T, Pietikanen M, Maenpaa T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI, 24(7), 971-987.
4. Carreras J, et al. (2025). Histological Image Classification Between Follicular Lymphoma and Reactive Lymphoid Tissue Using Deep Learning and XAI. Cancers, 17(15), 2428.

---

## 7. Sviluppi Futuri: Graph Neural Networks su Whole Slide Images

### Perche Delaunay e MST non sono stati usati in questo lavoro

Come documentato nella Sezione 1.1, le feature Delaunay e MST sono state rimosse per due ragioni concrete e interdipendenti:

1. **Incompatibilita con il modello tabulare (Fase 4):**
   Il Random Forest e XGBoost richiedono un vettore di numeri come input per ogni campione.
   Un grafo di Delaunay con ~150 nodi e ~450 archi e una struttura dati completamente diversa
   da un vettore — non puo entrare direttamente in un modello tabulare.
   Per usarlo, bisogna collassarlo in numeri (mean/std degli archi), e a quel punto
   il risultato e quasi identico al k-NN, con in piu i problemi di boundary effect.

2. **Boundary effects sulla patch 224x224 px (51.5 um):**
   I grafi spaziali costruiti su una singola micro-patch vengono troncati artificialmente
   ai quattro bordi. Nuclei biologicamente vicini ma in patch diverse risultano
   disconnessi nel grafo, generando descrittori spaziali distorti.

### Come potrebbero essere usati in un lavoro futuro

La strada corretta per sfruttare la potenza di Delaunay e MST e la seguente:

STEP 1: Operare su Whole Slide Images (WSI)
   - Campo visivo: ordine dei millimetri (vs 51.5 um della patch attuale)
   - Scala adeguata per catturare l-architettura macro-follicolare del FL
   - No boundary effects: il grafo si estende sull-intero linfonodo

STEP 2: Costruire il grafo di Delaunay su tutti i nuclei della WSI
   - Nodi: centroidi dei nuclei (x, y) in um
   - Archi: connessioni Delaunay con peso = distanza in um
   - Attributi dei nodi: feature morfometriche per singolo nucleo (area, circolarita, ecc.)

STEP 3: Addestrare una Graph Neural Network (GNN)
   - Architetture candidate: GraphSAGE, Graph Attention Network (GAT), GIN
   - La GNN apprende messaggi tra nodi vicini, propagando informazione
     attraverso la topologia del grafo
   - Classificazione finale: FL vs Reactive a livello di WSI (o di follicolo)

Questo approccio permetterebbe di catturare sia la morfologia dei singoli nuclei
(attributi dei nodi) che l-architettura spaziale globale del tessuto (struttura del grafo)
in un unico modello end-to-end, superando la limitazione di scala del presente lavoro.

Riferimenti per sviluppo futuro:
- Chen et al. (2021). Whole slide images based cancer survival prediction using
  attention guided deep multiple instance learning networks.
  Medical Image Analysis, 65, 101789.
- Hamilton WL, Ying R, Leskovec J. (2017). Inductive Representation Learning
  on Large Graphs (GraphSAGE). NeurIPS 2017.
- Veličković P, et al. (2018). Graph Attention Networks (GAT). ICLR 2018.
