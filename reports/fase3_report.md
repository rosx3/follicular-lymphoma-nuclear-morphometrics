# Report Fase 3 — Estrazione Biomarcatori Citomorfometrici, Spaziali e di Tessitura
### Tesi: Quantificazione Citomorfometrica e Spaziale per la Classificazione tra Linfoma Follicolare e Tessuto Reattivo
*Modulo: src/03_feature_extraction.py*
*Aggiornato: 19 agosto 2026 — Corretta discrepanza nel conteggio delle feature (vedi nota in Sezione 2)*

> **Piano operativo:** `reports/fase3_implementation_plan.md` — stato di avanzamento,
> decisioni metodologiche approvate (D1–D7) e task di implementazione.
> Questo report descrive *cosa* si misura e *perché*; il piano descrive *come* e *a che punto siamo*.

---

## Obiettivo

Tradurre le **94.042 maschere d-istanza nucleare** prodotte dalla Fase 2 e i **600 H-channel CLAHE** prodotti dalla Fase 1 in una matrice tabulare di **biomarcatori fisici e clinicamente interpretabili**, espressi in unita reali (um, um2) con calibrazione 1 px = 0.46 um (scanner Hamamatsu NanoZoomer S360, patch esportate a 200x = obiettivo 20x; derivazione in reports/fase1_report.md, revisione del 19 agosto 2026).

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
| knn5_dist_mean/std_um | Fortemente correlato con k=3 alla scala micro-locale della patch (103.0 um di lato). | Analisi scala spaziale dataset |
| Triangolazione di Delaunay (5 feature) | Boundary effects critici: il grafo viene troncato ai bordi della patch (224x224 px = 103.0 um). Delaunay/MST hanno validita su WSI (ordine mm), non su micro-patch. | Letteratura 2024: edge effects at patch boundary |
| Minimum Spanning Tree (4 feature) | Stessa motivazione Delaunay. mst_edge_length_mean e quasi equivalente a knn1_dist_mean su distribuzioni quasi-uniformi. | Letteratura 2024 + analisi matematica |
| glcm_correlation | Alta correlazione con glcm_homogeneity su immagini H&E normalizzate. | Haralick (1973) |
| glcm_dissimilarity | Fortemente correlata con glcm_contrast. | Haralick (1973) |
| lbp_mean, lbp_std | Solo l-entropia LBP e robusta e interpretabile come misura di complessita della micro-tessitura. | Ojala et al. (2002) |
| hchannel_skew | Ridondante con area_um2_skew e circularity_skew. Dopo normalizzazione Macenko la varianza cromatica e gia ridotta del 68.6%. | reports/fase1_report.md |
| Momenti Cromatici CIE-LAB (3 feature) | Completamente ridondanti post-normalizzazione Macenko: lo scopo di Macenko e portare tutte le patch nello stesso spazio cromatico. Misurare i colori dopo = misurare il rumore residuo. | reports/fase1_report.md + letteratura |

NOTA PER LA TESI (Sezione Discussione):
Le feature Delaunay e MST sono scientificamente valide per l-analisi della micro-architettura tissutale,
ma richiedono un campo visivo nell-ordine dei mm (Whole Slide Images). Su patch 224x224 px (103.0 um)
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

**Definizione operativa.** Per ogni nucleo si calcola la media delle distanze euclidee
ai suoi *k* vicini piu prossimi (centroidi in um, `scipy.spatial.KDTree`, self-match
escluso); la colonna `_mean` e la media di questo valore su tutti i nuclei della patch,
la colonna `_std` la sua deviazione standard. La precisazione serve perche "distanza
media ai 3 vicini" ammette due letture — media per nucleo poi aggregata, oppure pool
di tutte le distanze — che producono la stessa `_mean` ma `_std` diverse.

Con meno di *k+1* nuclei la statistica non e definita e la colonna vale `NaN`, non `0.0`:
uno zero verrebbe interpretato dal modello come nuclei sovrapposti, cioe densita massima,
l'opposto della situazione reale di una patch quasi vuota.

### 2.6 Tessitura Cromatinica H-channel (6 colonne)
| Feature | Descrizione | Fonte |
|---|---|---|
| glcm_contrast | Variazione intensita cromatina (dispersa=alta) | Haralick et al. (1973) |
| glcm_homogeneity | Uniformita cromatina (compatta=alta) | Haralick et al. (1973) |
| glcm_energy | Ordine/uniformita (ASM) | Haralick et al. (1973) |
| lbp_entropy | Complessita micro-tessitura cromatina | Ojala et al. (2002) |
| hchannel_mean | Intensita media ematossilina nella patch | Standard patologia digitale |
| hchannel_std | Variabilita cromatina intra-patch | Standard patologia digitale |

**Le sei feature sono calcolate sui soli pixel nucleari**, usando le maschere d'istanza
della Fase 2 come maschera binaria, non sull'intera patch. Calcolarle sull'intera patch
misurerebbe anche stroma, spazio inter-nucleare e sfondo, diluendo il segnale che si
intende quantificare: su 80 patch reali l'intensita media dell'ematossilina passa da
70.2 (intera patch) a 156.7 (soli nuclei), quindi la versione non mascherata sarebbe
dominata dallo sfondo.

Parametri (fissati in `src/03_feature_extraction.py` e replicati nel metadata JSON):

| Descrittore | Parametri |
|---|---|
| GLCM | 64 livelli di grigio, distanza 1 px, angoli 0/45/90/135 gradi mediati, matrice simmetrica |
| LBP | 8 punti, raggio 1 px, metodo `uniform` (10 bin), entropia di Shannon in base 2 |

*Nota implementativa sul mascheramento della GLCM.* `graycomatrix` non accetta maschere:
l'H-channel viene quantizzato su 63 livelli mappati su 1..63 riservando lo 0 allo sfondo,
e dopo il calcolo si scartano riga 0 e colonna 0, cioe tutte le coppie che coinvolgono
un pixel non nucleare. Lo scarto sposta gli indici di livello di 1, ma le tre proprieta
usate sono invarianti a questo shift: contrasto e omogeneita pesano i termini con (i-j),
dove uno spostamento costante di entrambi si annulla, e l'energia non dipende dagli
indici. La correlazione di Haralick, che userebbe indici assoluti e verrebbe distorta,
era gia esclusa dal set per ridondanza (1.1).

Se nella patch non esiste alcuna coppia di pixel nucleari adiacenti (nuclei ridotti a
pixel isolati), le quattro colonne di tessitura valgono `NaN`: `graycoprops`
restituirebbe altrimenti zeri indistinguibili da una tessitura perfettamente piatta.
Le due statistiche di intensita restano calcolabili, non richiedendo adiacenza.

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

### 3.4 Verifica di Sanita Dimensionale — ANOMALIA RISOLTA (calibrazione corretta)

Verifica condotta il 19 agosto 2026 durante l'implementazione delle distanze k-NN
(Task 1 del piano), su 30 patch di Ground Truth e 60 patch del dataset completo,
**prima** dell'estrazione finale.

#### Fatti misurati

| Grandezza | Valore misurato | Come |
|---|---|---|
| Nuclei rilevati dal Watershed vs Ground Truth | **0.90** (155.4 vs 174.1 per patch) | 30 patch GT, `gt_metadata.json` |
| Diametro equivalente medio dei nuclei (Watershed) | **2.48 um** | 60 patch |
| Diametro equivalente medio dei nuclei (Ground Truth) | **2.51 um** | 30 patch GT |
| Frazione di area nucleare della patch | **0.313** (min 0.162, max 0.514) | 60 patch |
| Distanza media al primo vicino | **~3.0 um** | 80 patch |

Due conseguenze immediate:

1. **Il Watershed sotto-rileva, non sovra-segmenta.** Produce il 90% dei nuclei della
   Ground Truth. Va corretta qualunque affermazione in senso opposto.
2. **L'anomalia dimensionale non e un artefatto della segmentazione.** Ground Truth e
   Watershed concordano sul diametro (2.51 vs 2.48 um): il valore anomalo e presente
   in entrambi, quindi non e prodotto dall'algoritmo di questo lavoro.

#### La contraddizione

Un nucleo linfoide ha in letteratura un diametro di circa 6-7 um, cioe un'area di
circa 33 um^2. Con i ~156 nuclei per patch effettivamente rilevati, l'area nucleare
totale sarebbe:

```
156 nuclei x 33 um^2 = 5.148 um^2   su un campo dichiarato di 2.654 um^2  ->  194%
```

Un valore geometricamente impossibile. Le tre grandezze — numero di nuclei, dimensione
attesa dei nuclei, area del campo visivo — non possono essere tutte corrette
simultaneamente.

#### Verifica sulla fonte primaria

Lo studio sorgente (Carreras et al., 2025, *Cancers* 17(15):2428, PMCID PMC12345699) e
stato consultato direttamente. Quanto dichiara nella sezione Materiali e Metodi:

- **Scanner:** NanoZoomer S360 digital slide scanner C13220-01 (Hamamatsu Photonics K.K.)
- **Esportazione delle immagini:** software NDP.view2, "converted into a jpeg file at
  200x magnification and 150 dpi" (Carreras et al., 2025)
- **Patch:** 224 x 224 x 3 px, dimensionate sull'input layer della rete
- Alcune figure dell'articolo sono esportate a 400x, 150 dpi

Quanto **non** dichiara, in nessun punto dell'articolo:

- Nessun valore di micron per pixel
- Nessuna dimensione fisica del campo visivo delle patch
- **Nessuna menzione di un obiettivo 40x** per le patch
- Nessuna informazione su sotto-campionamento o ridimensionamento

Il record Zenodo (DOI 10.5281/zenodo.15702609) non riporta alcun metadato tecnico. I file
JPEG contengono una densita JFIF di 96 dpi, valore di default generico scritto al momento
del ritaglio e non riconducibile ai 150 dpi dell'esportazione originale: non e utilizzabile
come evidenza.

**Conclusione della verifica: la calibrazione di 0.23 um/px non e supportata dalla fonte.**
Il valore 0.23 um/px corrisponde alla risoluzione nativa del NanoZoomer S360 con obiettivo
40x, cioe a un ingrandimento totale di 400x. Le patch, per esplicita dichiarazione degli
autori, sono state convertite a **200x**, non a 400x. Nella notazione istopatologica
convenzionale l'ingrandimento e riportato come prodotto obiettivo x oculare (10x), quindi
200x corrisponde a un **obiettivo 20x** — la meta della risoluzione nativa, cioe
**0.46 um/px**. Coerentemente, le figure a 400x dell'articolo corrispondono all'obiettivo
40x nativo.

#### Confronto delle calibrazioni candidate

La frazione di area nucleare (0.313), il diametro in pixel (10.78 px) e il conteggio
(154 nuclei/patch) sono **rapporti di conteggi di pixel e non dipendono dalla
calibrazione**: restano validi qualunque sia il valore corretto e permettono di
discriminare fra le alternative.

| um/px | Lettura di "200x" | Campo visivo | Diametro nucleare | Densita nucleare |
|---|---|---|---|---|
| 0.230 | *precedente* — obiettivo 40x (= 400x) | 51.5 um | 2.48 um | 58.019 /mm2 |
| **0.460** | **ADOTTATA** — obiettivo 20x (= 200x) | **103.0 um** | **4.96 um** | **14.505 /mm2** |
| 0.847 | 200x @150dpi come scala di stampa | 189.7 um | 9.13 um | 4.282 /mm2 |

Riferimenti di letteratura per il tessuto linfoide: diametro nucleare 6-12 um
(centrociti e centroblasti), densita nucleare dell'ordine di 10.000-20.000 nuclei/mm2.

**La densita nucleare discrimina nettamente.** A 0.23 um/px risulta 58.019 nuclei/mm2,
un valore fisicamente impossibile. A 0.847 um/px risulta 4.282 nuclei/mm2, troppo rado
per un centro germinativo. A **0.46 um/px** risulta 14.505 nuclei/mm2, in pieno intervallo
di letteratura.

Il diametro conferma: a 0.46 um/px la misura e 4.96 um, che corretta per la sotto-copertura
del Watershed (Dice 0.637 rispetto alla Ground Truth, quindi il diametro reale e circa
1/sqrt(0.637) volte quello misurato) porta a **~6.2 um**, valore atteso per un linfocita.

**Valore adottato: 0.46 um/px**, esattamente il doppio del precedente.
Resta una deduzione, non un dato dichiarato: gli autori non pubblicano la scala. Le due
verifiche indipendenti — densita nucleare e diametro corretto — convergono entrambe su
questo valore, ma una conferma definitiva richiede di contattare gli autori o di misurare
una struttura di dimensione nota. La spiegazione discorsiva completa, adatta a essere
riportata nella tesi, e in reports/fase1_report.md.

#### Cosa e coinvolto e cosa no

Se l'ipotesi fosse confermata, la calibrazione agirebbe come un fattore di scala globale:
lineare sulle lunghezze, quadratico sulle aree.

**Non coinvolte** (adimensionali o rapporti di pixel): `n_nuclei`,
`nuclear_area_fraction`, `circularity`, `eccentricity`, `solidity`, `aspect_ratio`,
tutte le colonne `_cv`, e le 6 feature di tessitura.

**Coinvolte** (tutte le colonne in um e um^2): `area_um2_*`, `perimeter_um_*`,
`major_axis_um_*`, `minor_axis_um_*`, `area_top10_mean_um2`, `area_top10_short_axis_um`,
`nuclear_density_per_1000um2` e le 4 colonne k-NN.

Se il valore corretto fosse 0.46 um/px, la conversione sarebbe: **lunghezze x2**,
**aree x4**, `nuclear_density_per_1000um2` **/4**. Nessuna ri-esecuzione della
segmentazione sarebbe necessaria — basta cambiare `MICRONS_PER_PIXEL` e rigenerare
i CSV della Fase 3.

**Impatto sui risultati.** Un fattore di scala globale e una trasformazione monotona
identica su tutte le patch: **non** altera l'ordinamento fra patch, **non** cambia la
significativita dei test di separabilita FL vs REACTIVE (Sezione 3.3), e **non** cambia
le prestazioni dei modelli ad albero della Fase 4, che sono invarianti a riscalature
monotone delle feature.

Cio che invaliderebbe e l'**interpretazione clinica assoluta** e il confronto diretto
con le soglie dimensionali di Iwamoto et al. (2024) — cioe proprio uno dei punti di forza
dichiarati dell'approccio white-box in unita fisiche reali. Per questo la verifica va
fatta prima di consegnare, ma non blocca l'esecuzione della Fase 3.


#### Stato: revisione applicata il 19 agosto 2026

La calibrazione e stata portata a **0.46 um/px** in tutto il progetto. I valori misurati
riportati sopra (diametro 2.48 um, densita 58.019/mm2, campo 51.5 um) sono quelli
osservati **con la calibrazione precedente** e sono conservati perche documentano come
l'anomalia e stata individuata.

Interventi effettuati:

- Costante centralizzata in `src/calibration.py`, unica fonte di verita, con la
  provenienza documentata. I moduli 01, 02, 03 e `run_pipeline.py` la importano da li;
  in precedenza il valore era duplicato in quattro moduli piu due letterali
  non collegati alla costante in `run_pipeline.py`.
- Ricalcolate le colonne in unita fisiche di `data/fase2_segmentation/centroids_all.csv`
  (94.042 righe: `centroid_x_um`, `centroid_y_um`, `area_um2`), ricomputandole dalle
  colonne in pixel anziche riscalando i micron, per non accumulare arrotondamenti.
- Aggiornati `preprocessing_metadata.json` e `gt_metadata.json`.
- Aggiornati i valori derivati nei report: tile CLAHE 28x28 px (12.9 um), distanza
  minima Watershed 12 px (5.5 um), area massima nucleare 2500 px (529 um2), diametro
  Cellpose 22.0 px (10.1 um).
- Aggiunti test di regressione che impediscono di ridefinire o riscrivere la scala
  fuori da `src/calibration.py` (`tests/test_calibration.py`).

Le maschere di segmentazione non sono state rigenerate: la Fase 2 lavora in pixel e non
e influenzata dalla calibrazione.

---

## 4. Anteprime Grafiche

[Immagini da inserire dopo esecuzione]

---

## 5. Dipendenze e Riproducibilita

### 5.1 Librerie impiegate

| Libreria | Uso |
|---|---|
| skimage.measure.regionprops | Morfometria d'istanza per singolo nucleo |
| numpy | Aggregazioni statistiche (mean/std/cv) |
| scipy.stats.skew | Asimmetria delle distribuzioni intra-patch |
| scipy.spatial.KDTree | Distanze k-NN sui centroidi |
| skimage.feature.graycomatrix / graycoprops | GLCM e proprieta di Haralick |
| skimage.feature.local_binary_pattern | LBP per l'entropia di micro-tessitura |
| opencv-python | Lettura di maschere 16-bit e H-channel |

### 5.2 File dei metadati

I due CSV non bastano a rendere riproducibile la Fase 3: la quantizzazione della GLCM,
il raggio dell'LBP, i valori di *k* e la calibrazione spaziale determinano i numeri
estratti ma **non sono desumibili dai dati**. Sono quindi registrati insieme all'output
in `data/fase3_features/feature_extraction_metadata.json`, generato automaticamente a
ogni esecuzione.

| Blocco | Contenuto |
|---|---|
| `calibrazione` | um/px, area del pixel, lato e area della patch, provenienza del valore |
| `conteggi` | patch processate, patch in errore, nuclei totali |
| `feature` | numero di feature e metadati, elenco completo e ordinato delle 50 colonne |
| `parametri_glcm` | livelli, distanze, angoli, simmetria, proprieta estratte, mascheramento |
| `parametri_lbp` | punti, raggio, metodo, numero di bin, base dell'entropia, mascheramento |
| `parametri_knn` | valori di *k*, metrica, esclusione del self-match, valore sui casi non definiti |
| `decisioni` | riferimento sintetico alle decisioni metodologiche D1, D2, D3, D7 |
| `ambiente` | versioni di Python, numpy, scipy, scikit-image, opencv |
| `tempo_esecuzione_s` | durata dell'estrazione |

L'elenco delle colonne nel file **non e scritto a mano**: proviene dalla stessa costante
che genera l'intestazione del CSV, quindi i due non possono divergere. Un test di
integrazione lo verifica confrontando le due liste.

Le versioni delle librerie sono registrate perche i default degli algoritmi cambiano fra
release: un valore di `glcm_energy` non e confrontabile fra versioni diverse di
scikit-image senza sapere quale sia stata usata.

### 5.3 Parametri fissati

| Descrittore | Parametri |
|---|---|
| GLCM | 64 livelli, distanza 1 px, angoli 0/45/90/135 gradi mediati, matrice simmetrica, ristretta ai pixel nucleari |
| LBP | 8 punti, raggio 1 px, metodo `uniform` (10 bin), entropia di Shannon in base 2, istogramma sui soli pixel nucleari |
| k-NN | k = 1 e 3, distanza euclidea sui centroidi in um, self-match escluso, `NaN` con meno di k+1 nuclei |
| Calibrazione | 1 px = 0.46 um (vedi Sezione 3.4 e `reports/fase1_report.md`) |

### 5.4 Come riprodurre

```bash
pip install -r requirements.txt        # versioni fissate con ==
python src/run_pipeline.py --fase 3    # rigenera CSV, anteprima e metadati
python -m pytest tests/ -q             # verifica la suite
```

La Fase 3 e deterministica: non usa campionamenti casuali e, a parita di input e di
versioni, produce byte identici.

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

2. **Boundary effects sulla patch 224x224 px (103.0 um):**
   I grafi spaziali costruiti su una singola micro-patch vengono troncati artificialmente
   ai quattro bordi. Nuclei biologicamente vicini ma in patch diverse risultano
   disconnessi nel grafo, generando descrittori spaziali distorti.

### Come potrebbero essere usati in un lavoro futuro

La strada corretta per sfruttare la potenza di Delaunay e MST e la seguente:

STEP 1: Operare su Whole Slide Images (WSI)
   - Campo visivo: ordine dei millimetri (vs 103.0 um della patch attuale)
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
