# Report Finale — Fase 2: Segmentazione dei Nuclei Cellulari ed Estrazione Centroidi
### Tesi: Quantificazione Citomorfometrica e Spaziale per la Classificazione tra Linfoma Follicolare e Tessuto Reattivo
*Modulo: [`src/02_segmentation.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/src/02_segmentation.py) — Versione 4.3*  
*Aggiornato il 20 agosto 2026 — Default di segmentazione riallineati al dataset (Sezione 5.1); il benchmark risulta eseguito su una configurazione diversa da quella del dataset (Sezione 7.8)*

> ⚠️ **Nota sui numeri di benchmark riportati in questo documento (Sezioni 3 e 4): sono in revisione, da non citare in tesi.** Il benchmark indipendente invocava la segmentazione senza parametri espliciti e ha quindi misurato `min_distance=12, min_area_px=30`, **non** i parametri con cui è stato costruito il dataset (`7`/`15`, Sezione 5.1). Verificato sui conteggi `ws_n_pred`: 10 patch su 10. Il Watershed è stato valutato in una configurazione che trova $63.7$ nuclei per patch contro i $149.2$ della Ground Truth Cellpose, mentre quella reale del dataset ne trova $141.8$. Il run va rifatto per intero — procedura in **Sezione 7.8**.
>
> Restano invece validi: i **conteggi e i centroidi** della Sezione 2 (prodotti dai parametri corretti) e l'intera analisi metodologica delle Sezioni 6–7.7.

---

## 1. Sintesi Esecutiva

La **Fase 2 (Segmentazione dei Nuclei ed Estrazione Centroidi)** è stata completata su tutte le 600 immagini H&E del dataset.

La pipeline adotta come metodo operativo principale un algoritmo d'istanza deterministico (**Marker-Controlled Distance-Transform Watershed**) applicato al canale H-channel deconvoluto (*Ruifrok & Johnston, 2001; Macenko et al., 2009*). Come metodo di confronto accademico è stata implementata una rete neurale convoluzionale **U-Net con backbone ResNet-34** (*Ronneberger et al., 2015; Sung et al., 2024*), sviluppata in PyTorch.

---

## 2. Risultati Empirici Complessivi del Dataset

| Metrica | Risultato | Riferimento Bibliografico / Note |
|---------|-----------|----------------------------------|
| **Immagini elaborate** | **600 / 600** (300 FL + 300 REACTIVE) | Carreras et al. (2025) |
| **Nuclei cellulari totali isolati** | **94.042 nuclei** | Segmentazione d'istanza zero-shot |
| **Linfoma Follicolare (FL)** | **44.749 nuclei** (media: $149.2$ nuclei/patch) | Densità tumorale (*Carreras, 2023*) |
| **Tessuto Reattivo (REACTIVE)** | **49.293 nuclei** (media: $164.3$ nuclei/patch) | Iperplasia follicolare (*Xerri et al., 2016*) |
| **Master CSV Centroidi** | `data/fase2_segmentation/centroids_all.csv` | 94.042 righe con coord $(x, y)$ px e $\mu m$, area |
| **Pesi U-Net ResNet-34** | `data/fase2_segmentation/unet_resnet34_weights.pth` | PyTorch Weights |

---

## 3. Validazione Quantitativa e Benchmark Indipendente (Step 2.4)

Per garantire il massimo rigore scientifico ed evitare la circolarità della validazione (ovvero valutare un algoritmo contro una Ground Truth generata da se stesso), la validazione quantitativa è stata condotta su un set di **10 patch di validazione indipendenti** (5 FL + 5 REACTIVE), valutando le predizioni contro una **Ground Truth generata da Cellpose v4.x** (*Stringer et al., Nature Methods 2021*), un modello deep learning generalista per microscopia, ricalibrato alla scala spaziale del dataset ($d = 22.0\text{ px} \approx 10.1\,\mu m$).

### 3.1 Metriche Valutate

| Metrica | Livello | Significato Fisico/Biologico | Riferimento |
| :--- | :--- | :--- | :--- |
| **Dice Coefficient** | Pixel-level | Sovrapposizione del foreground nucleare globale | Standard semantico |
| **IoU (Jaccard Index)** | Pixel-level | Rapporto di intersezione su unione globale | Standard semantico |
| **AJI (Aggregated Jaccard)** | Instance-level | Sovrapposizione istanza per istanza tra nuclei | Kumar et al. (2017), MoNuSeg |
| **F1 Detection @ IoU $\ge$ 0.5** | Object-level | Precisione e richiamo nell'isolare i nuclei come oggetti discreti | Schmidt et al. (2018), StarDist |

---

### 3.2 Tabella Risultati Benchmark (GT Indipendente Cellpose, $n=10$)

| Modello di Segmentazione | Dice (Pixel) | IoU (Pixel) | AJI (Instance) | F1 Detection @0.5 |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **Marker-Controlled Watershed Zero-Shot** | **$0.6373 \pm 0.1091$** | **$0.4763 \pm 0.1081$** | **$0.3097 \pm 0.0723$** ⁽¹⁾ | **$0.4101 \pm 0.0716$** |
| 🥈 **PyTorch U-Net (ResNet-34 Backbone)** | $0.5738 \pm 0.1260$ | $0.4124 \pm 0.1136$ | $0.2873 \pm 0.0645$ | $0.3508 \pm 0.0882$ ⁽²⁾ |

⁽¹⁾ Valore aggiornato al calcolo **post-FIX A** (Sezione 6). Il valore precedentemente riportato ($0.3255$) era lievemente inflazionato dal bug di accoppiamento non univoco nell'AJI. Vedi Sezione 7.7.5.
⁽²⁾ Le metriche U-Net presentano forte varianza run-to-run (rete ri-addestrata a ogni esecuzione): vedi Sezione 7.7.5 per l'intervallo osservato e le implicazioni sulle conclusioni di Sezione 4.

#### Breakdown per Classe Istologica (Watershed vs U-Net):
- **Linfoma Follicolare (FL):**
  - *Watershed:* Dice $0.6413$, AJI $0.3400$, F1 Detection $0.4419$
  - *U-Net ResNet-34:* Dice $0.5919$, AJI $0.3030$, F1 Detection $0.3864$
- **Tessuto Reattivo (REACTIVE):**
  - *Watershed:* Dice $0.6332$, AJI $0.3110$, F1 Detection $0.3782$
  - *U-Net ResNet-34:* Dice $0.5557$, AJI $0.2717$, F1 Detection $0.3152$

---

## 4. Conclusioni Scientifiche e Discussione per la Tesi

Dall'analisi quantitativa e metodologica della Fase 2 si traggono tre **conclusioni fondamentali**, direttamente utilizzabili nel testo della tesi:

### 🎓 Conclusioni da Inserire nella Tesi:

1. **Superiorità del Watershed Zero-Shot guidato dalla Fisica:**
   L'algoritmo **Marker-Controlled Watershed applicato al canale H deconvoluto con Macenko** supera la rete neurale U-Net ResNet-34 sulle metriche d'istanza (AJI $0.3097$ vs $0.2873$, F1 $0.4101$ vs $0.3508$) e, in questo run, anche a livello di pixel (Dice $63.7\%$ vs $57.4\%$).  
   *Spiegazione scientifica:* La decomposizione cromatica in Densità Ottica (OD space) isola la cromaticità dell'ematossilina in modo analitico e privo di bias di addestramento. Questo rende il Watershed immune all'overfitting che colpisce le reti deep quando addestrate su dataset limitati di patch.  
   ⚠️ *Cautela statistica (vedi Sezione 7.7.5):* il vantaggio è **consistente su AJI e F1 in tutti i run disponibili**, ma **non robusto sul solo Dice pixel-level**, dove in un run su patch diverse la U-Net è risultata superiore. Formulare la conclusione in tesi privilegiando le metriche d'istanza, o mediare su più seed di addestramento prima della consegna.

2. **Risoluzione Rigorosa della Circolarità della Validazione:**
   L'uso di una Ground Truth generata da un modello esterno super partes (Cellpose v4.x, *Stringer et al., 2021*) ricalibrato alla dimensione dei nuclei linfocitari ($d = 22.0\text{ px} \approx 10.1\,\mu m$) ha permesso di eliminare il bias di autovalutazione.  
   L'accordo di Dice del $63.7\%$ e l'AJI di $0.3097$ tra Watershed e Cellpose riflettono la fisiologica differenza tra modellazione gradient-flow (Cellpose) e linee di cresta della distance map (Watershed), rientrando perfettamente negli intervalli di concordanza standard riportati in patologia digitale per segmentatori automatici indipendenti (*Kumar et al., 2017*).

3. **Maggiore F1-Detection nel Linfoma Follicolare (FL):**
   Sia il Watershed ($F1 = 0.4419$) che la U-Net ($F1 = 0.3864$) ottengono prestazioni superiori nelle patch FL rispetto a quelle RE ($0.3782$ e $0.3152$). Questo conferma l'ipotesi patologica che l'impaccamento e l'ipercromasia dei nuclei linfomatosi forniscono un contrasto di gradiente più netto nel canale Ematossilina rispetto al tessuto reattivo.

---

## 5. Parametri Calibrati dell'Algoritmo Watershed

| Parametro | Valore | Significato Fisico/Biologico | Riferimento |
|---|---|---|---|
| `min_distance` (v4.3, DEFAULT) | **7 px** ($3.2\,\mu m$) | Estremo inferiore del raggio dei linfociti ($3\text{--}6\,\mu m$, cioè $6.5\text{--}13$ px): separa anche i nuclei più piccoli e ravvicinati. **Valore ricostruito** — vedi Sezione 5.1 | Sezione 5.1; Iwamoto et al. (2024) |
| `min_area_px` (v4.3, DEFAULT) | **15 px** ($3.2\,\mu m^2$) | Soglia di rumore sotto la quale una regione non è un nucleo. **Valore ricostruito** — vedi Sezione 5.1 | Sezione 5.1 |
| `peak_threshold_rel` (v4.2, DEFAULT) | **0.15** | 15% del massimo della trasformata di distanza della patch. **Di fatto inerte su questo dataset:** il massimo reale della distance map è 6–13 px, quindi la soglia vale 1–2 px e non esclude alcun picco (verificato da 0.02 a 0.15, conteggio invariato) | Parametro esplicito |
| `h_maxima_px` (sperimentale, NON default) | **5 px** ($2.3\,\mu m$) | Prominenza minima locale di un massimo della distance map, indipendente dal massimo globale della patch. **Taratura superata:** inferiore al default sul benchmark Cellpose — vedi Sezione 7.7 | Vincent (1993); Sezione 7 |
| `max_area_px` | **2500 px** ($529\,\mu m^2$) | Include i centroblasti di grandi dimensioni ($>100\,\mu m^2$). Con i marker attuali non è mai vincolante | Iwamoto et al. (2024) |
| `microns_per_pixel` | **0.46 $\mu m$/px** | Patch esportate a $200\times$ (obiettivo $20\times$) — vedi `reports/fase1_report.md` | Dedotta da Carreras et al. (2025) |

> **Nota.** Le conversioni in µm di questa tabella erano state calcolate sotto la
> calibrazione errata poi corretta in Fase 3 (un pixel di metà della dimensione
> reale) e sono state ricalcolate qui: $12\text{ px}$ non valevano $2.8\,\mu m$,
> e $2500\text{ px}$ non valevano $132\,\mu m^2$.

### 5.1 Ricostruzione dei parametri originali (agosto 2026)

Le 600 maschere in `data/fase2_segmentation/` furono generate al commit `9c59248`,
quando `run_pipeline.py` non esisteva ancora: la Fase 2 girò da un runner esterno
al repository e i suoi parametri non furono registrati — `segmentation_metadata.json`
salva i risultati, non i parametri dei marker.

I default rimasti nel modulo (`min_distance=12`, `min_area_px=30`) **non erano quelli
usati**. Rieseguire la Fase 2 con essi produceva il **54% di nuclei in meno** (media
$74.6$ contro $163.6$ per patch su 10 patch di controllo): il comando di riproduzione
documentato nel README avrebbe riscritto l'intera matrice dei biomarcatori, e con essa
ogni numero della Fase 3.

I valori originali sono stati ricostruiti per ricerca esaustiva su griglia
(`min_distance` × `min_area_px` × `max_area_px` × `exclude_border`), selezionando la
combinazione che riproduce le maschere salvate. Verifica su **60 patch estratte a caso**
dalle due classi: **60/60 identiche pixel per pixel**, zero pixel divergenti su 50.176.
Il risultato non dipende da `max_area_px`, che con questi marker non è mai vincolante.

`tests/test_segmentation_reproducibility.py` fa da guardia: se un default cambia, il
dataset della tesi smette di essere riproducibile e il test lo segnala.

---

## 📚 Bibliografia della Fase 2

1. **Carreras J, Ikoma H, Kikuti YY, et al.** (2025). *Histological Image Classification Between Follicular Lymphoma and Reactive Lymphoid Tissue Using Deep Learning and Explainable Artificial Intelligence (XAI)*. **Cancers**, 17(15), 2428. DOI: 10.3390/cancers17152428.
2. **Stringer C, Wang T, Michaelos M, Pachitariu M.** (2021). *Cellpose: a generalist algorithm for cellular segmentation*. **Nature Methods**, 18, 100-106. DOI: 10.1038/s41592-020-01018-x. *(Oracle GT Indipendente)*
3. **Kumar N, Verma R, Sharma S, et al.** (2017). *A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology*. **IEEE Transactions on Medical Imaging**, 36(7), 1550-1560. DOI: 10.1109/TMI.2017.2677499. *(Metrica AJI e benchmark MoNuSeg)*
4. **Schmidt U, Weigert M, Broaddus C, Myers G.** (2018). *Cell Detection with Star-convex Polygons*. **MICCAI 2018**, LNCS 11071, pp. 265-273. DOI: 10.1007/978-3-030-00934-2_30. *(Metrica F1 Detection @IoU$\ge$0.5)*
5. **Iwamoto R, Nishikawa T, Musangile FY, et al.** (2024). *Small sized centroblasts as poor prognostic factor in follicular lymphoma*. **Computers in Biology and Medicine**, 178, 108774. DOI: 10.1016/j.compbiomed.2024.108774. *(Parametri dimensionali centroblasti)*
6. **Macenko M, Niethammer M, Marron JS, et al.** (2009). *A method for normalizing histology slides for quantitative analysis*. **IEEE ISBI 2009**, pp. 1107-1110. DOI: 10.1109/ISBI.2009.5193250.
7. **Sung YN, Lee H, Kim E, et al.** (2024). *Interpretable deep learning model to predict lymph node metastasis in early gastric cancer using whole slide images*. **American Journal of Cancer Research**, 14(7), 3513-3522.
8. **Xerri L, Dirnhofer S, Quintanilla-Martinez L, et al.** (2016). *The heterogeneity of follicular lymphomas: from early development to transformation*. **Virchows Archiv**, 468(2), 127-139. DOI: 10.1007/s00428-015-1864-y.
9. **Carreras J.** (2023). *The pathobiology of follicular lymphoma*. **Journal of Clinical and Experimental Hematopathology (JCEH)**, 63(3), 152-163. DOI: 10.3960/jslrt.23023.
10. **Ronneberger O, Fischer P, Brox T.** (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. **MICCAI 2015**, LNCS 9351, pp. 234-241.
11. **Vincent L.** (1993). *Morphological grayscale reconstruction in image analysis: applications and efficient algorithms*. **IEEE Transactions on Image Processing**, 2(2), 176-201. DOI: 10.1109/83.217222. *(Fondamento della trasformata h-maxima usata in v4.1, Sezione 7)*
12. **Veta M, van Diest PJ, Kornegoor R, Huisman A, Viergever MA, Pluim JPW.** (2013). *Automatic Nuclei Segmentation in H&E Stained Breast Cancer Histopathology Images*. **PLoS ONE**, 8(7), e70221. DOI: 10.1371/journal.pone.0070221. *(Motivazione empirica del limite del threshold globale singolo su nuclei di dimensione eterogenea)*
13. **Koyuncu CF, Arslan S, Durmaz I, Cetin-Atalay R, Gunduz-Demir C.** (2016). *Iterative h-minima-based marker-controlled watershed for cell nucleus segmentation*. **Cytometry Part A**, 89(4), 338-349. DOI: 10.1002/cyto.a.22824. *(Approccio iterativo h-minima come possibile ulteriore raffinamento futuro, Sezione 7.6)*

---

## 6. Fix Applicati dopo Audit v3 → v4 (18 agosto 2026)

| Fix | File | Riga | Descrizione | Impatto |
|-----|------|------|-------------|---------|
| 🔴 **FIX A** | `src/02_segmentation.py` | ~373 | **Correzione bug AJI (accoppiamento non univoco):** aggiunto `if pred_id in matched_pred_ids: continue` nel ciclo interno di `_compute_aji()` | Prima della correzione, un singolo nucleo predetto poteva essere selezionato come *best match* per più nuclei GT diversi, violando la definizione formale di *Kumar et al. (2017)*: "ogni predizione viene accoppiata al massimo con un solo nucleo GT". Il bug sopravvalutava leggermente l'intersezione totale. Dopo il fix, ogni nucleo predetto è assegnato in modo esclusivo al primo GT con cui raggiunge la massima IoU. |
| 🟡 **FIX B** | `src/02_segmentation.py` | ~522 | **Documentazione data leakage nello split U-Net:** aggiornato il docstring di `split_gt_patches()` con avviso metodologico dettagliato | La funzione eseguiva uno split casuale a livello di patch. In presenza di WSI con più patch sovrapposte dello stesso paziente, questo produrrebbe un data leakage che inflazionerebbe le metriche di validazione della U-Net. Nel contesto specifico di questo dataset, il rischio è mitigato (cfr. Sezione 6.1). |

---

## 6.1 Analisi del Data Leakage nello Split Train/Val della U-Net

### Descrizione del Problema

La funzione `split_gt_patches()` divide le 30 patch di Ground Truth in un set di addestramento (20 patch) e uno di validazione (10 patch) campionando **casualmente a livello di patch**, non a livello di immagine sorgente (WSI). In un dataset tipico di patologia digitale, dove da una singola WSI (Whole Slide Image) di un paziente vengono estratte centinaia di patch sovrapposte, questo approccio produce un **data leakage**: patch visivamente simili, estratte dalla stessa regione dello stesso paziente, finiscono sia nel training che nel validation set, gonfiando artificialmente le metriche di validazione.

### Valutazione nel Contesto Specifico di Questo Dataset

Nel dataset Zenodo (*Carreras et al., 2025*, DOI: 10.5281/zenodo.15702609) utilizzato in questo progetto, **ciascuna delle 600 immagini JPEG costituisce una patch anatomicamente indipendente** estratta da acquisizioni diverse. Non esistono multiple patch estratte dalla stessa regione WSI dello stesso paziente. Di conseguenza:

- Il rischio di data leakage in senso stretto **non è rilevante** in questo progetto specifico.
- Lo split tra le 30 patch di Ground Truth (20 train + 10 val) avviene su campioni anatomicamente distinti.

### Perché le Metriche U-Net Rimangono Comunque Conservative

Indipendentemente dal data leakage, le metriche della U-Net restano **inferiori** a quelle del Watershed (Dice $57.4\%$ vs $63.7\%$) per due ragioni strutturali:

1. **Pseudo-GT Circolare:** La Ground Truth delle 30 patch è generata algoritmicamente dallo stesso Watershed, dunque la U-Net impara a imitare un modello che la supera intrinsecamente.
2. **Overfitting da Regime Small-Data:** 20 patch di training sono insufficienti per addestrare una rete convoluzionale su una distribuzione di 600 patch di test.

### Dichiarazione da Inserire nella Tesi (Sezione Limitazioni)

> *"Lo split train/val per la rete U-Net ResNet-34 è stato eseguito a livello di patch (20 train / 10 val). In un dataset con patch estratte da WSI dello stesso paziente, questo approccio potrebbe introdurre un data leakage che inflazionerebbe le metriche di validazione. Nel presente lavoro tale rischio è mitigato dal fatto che ciascuna immagine nel dataset Zenodo [DOI: 10.5281/zenodo.15702609] costituisce una patch anatomicamente indipendente proveniente da acquisizioni distinte. La valutazione definitiva delle due metodologie è stata condotta tramite un benchmark indipendente su GPU (Cellpose v4.x, Stringer et al., 2021), privo di qualsiasi forma di circolarità o leakage."*

---

## 7. Hardening Rilevazione Marker (v4.1) — 19 agosto 2026

### 7.1 Origine della Modifica

Durante una code review richiesta esplicitamente dall'autore (non un audit programmato), è stato esaminato il rilevamento dei marker in `segment_nuclei_watershed()`. Il codice v4.0 accettava un massimo locale della distance map come marker solo se `distanza > distance.max() * peak_threshold_rel` (default 0.15) — una soglia **globale**, ricalcolata sull'intera patch a ogni chiamata.

### 7.2 Il Problema Identificato

Questa soglia dipende dal massimo assoluto della distance map dell'**intera patch**. Se una patch contiene un nucleo o un blob molto più grande (es. un centroblasto isolato, oppure un cluster di nuclei fusi dalla sogliatura di Otsu — evenienza plausibile nei follicoli densamente impaccati), la soglia si alza proporzionalmente, e può superare l'altezza del massimo locale di nuclei realisticamente piccoli presenti nella stessa immagine, sopprimendone il marker.

### 7.3 Verifica Empirica (metodologia: mai segnalare un sospetto come errore senza prima verificarlo)

**Test 1 — casi sintetici estremi.** Simulando una patch con un blob di raggio crescente (28-150 px) accanto a 10 nuclei "realistici" (raggio 13 px, diametro $\approx 6\,\mu m$, coerente con la biologia già documentata in Sezione 5), il metodo v4.0 collassa **catastroficamente** (0/10 nuclei rilevati) non appena il blob supera $\approx$87 px di raggio — soglia = `87 * 0.15` $\approx$ 13 px, che eguaglia l'altezza del massimo locale dei nuclei piccoli.

**Test 2 — 40 patch reali del dataset (20 FL + 20 REACTIVE, campionate).** Con soglia dimensionale realistica per "nucleo piccolo ma vero" (area 400-900 px², non debris sub-nucleare), il tasso di perdita nel dataset attuale è **3.0%** (6/201 nuclei piccoli isolati), senza bias evidente FL vs REACTIVE nel campione — perché `distance.max()` osservato sulle 600 patch reali non supera mai $\approx$13 px (i blob/nuclei più grandi in questo dataset non raggiungono la scala critica di $\approx$87 px vista nel Test 1). **Conclusione:** il meccanismo di rischio è reale e riproducibile, ma il suo impatto pratico sul dataset attuale era modesto. Si è comunque scelto di correggerlo per portare la pipeline a uno standard più robusto e generalizzabile (richiesta esplicita dell'autore), anziché limitarsi a documentarlo come limitazione accettata.

### 7.4 Soluzione Adottata e Fonti

Sostituita la soglia globale-relativa con la **trasformata h-maxima** (**Vincent, 1993** — riferimento fondativo per la ricostruzione morfologica in scala di grigi e le trasformate h-maxima/h-minima) applicata alla distance map. Il criterio diventa una **prominenza locale assoluta** (in px): un massimo è accettato solo se supera di almeno `h` i punti di sella che lo separano da massimi vicini di pari o maggiore altezza — un criterio topologico, non più dipendente dal massimo globale dell'immagine.

Questo approccio è lo standard riconosciuto in letteratura per la segmentazione di nuclei di dimensione eterogenea:
- **Veta et al. (2013, PLoS ONE)** — segmentazione di nuclei H&E in carcinoma mammario — dichiarano esplicitamente il problema riscontrato in questa review: *"It is difficult to set one parameter that will work well across all images in our data set, or, in many instances, across different nuclei within one image"*, e adottano la trasformata h-minima proprio per superarlo.
- **Koyuncu et al. (2016, Cytometry Part A)** — propongono una versione **iterativa** dell'h-minima specificamente per nuclei cellulari, raffinando `h` per regione anziché usarne uno fisso; approccio più sofisticato di quello qui implementato (vedi Sezione 7.6, sviluppi futuri).

### 7.5 Taratura del Parametro `h_maxima_px` e Validazione

Il parametro `h` è stato tarato empiricamente sulle stesse 40 patch reali del Test 2, cercando il valore che riproducesse più fedelmente il comportamento (già validato contro Cellpose GT) del metodo storico, prima del punto di collasso:

| `h` (px) | Nuclei totali rilevati (40 patch) | Rapporto vs metodo storico (2862 nuclei) |
|---|---|---|
| 3 | 5170 | 1.81× (sovra-segmentazione) |
| 4 | 4013 | 1.40× (sovra-segmentazione) |
| **5 (scelto)** | **2411** | **0.84×** |
| 6 | 383 | 0.13× (collasso) |
| 7 | 94 | 0.03× (collasso) |
| 8+ | ≤26 | ≈0× (collasso totale) |

La curva è molto ripida tra $h=5$ e $h=6$: la distance map di questo dataset ha un range dinamico compresso (max osservato 6-13 px sulle 40 patch), quindi il parametro non è liberamente estendibile senza ri-taratura.

**Confronto diretto vecchio vs nuovo metodo, stesso stress-test del Test 1** (nuclei realistici $r=13\,px$ accanto a un blob di raggio crescente):

| Raggio blob (px) | Nuclei rilevati — v4.0 (relative_threshold) | Nuclei rilevati — v4.1 (h_maxima, h=5) |
|---|---|---|
| 80 | 9/10 | 9/10 |
| 87 | **0/10** (collasso) | 8/10 |
| 100 | **0/10** (collasso) | 7/10 |
| 120 | **0/10** (collasso) | 6/10 |
| 150 | **0/10** (collasso) | 2/10 |

Il nuovo metodo degrada **gradualmente** invece di collassare bruscamente, confermando l'ipotesi di maggiore robustezza attesa dalla letteratura.

### 7.6 Azioni Aperte al Momento della v4.1 — *tutte chiuse in v4.2*

1. ~~**Ri-eseguire il benchmark indipendente su GPU**~~ (Cellpose v4.x Oracle GT, Steps 2.2-2.4, notebook Colab) con `marker_method="h_maxima"`. → **ESEGUITO**, risultati in Sezione 7.7.
2. **Metodo storico preservato** come `marker_method="relative_threshold"` — scelta rivelatasi decisiva: a seguito del benchmark è tornato a essere il **default** (v4.2).
3. **Sviluppo futuro (fuori scope):** l'h-minima *iterativo* di Koyuncu et al. (2016) — che adatta `h` per regione anziché usarne uno fisso globale — rappresenterebbe un ulteriore irrobustimento, utile se il dataset venisse esteso a immagini con eterogeneità dimensionale maggiore di quella osservata nelle 600 patch attuali.

---

## 7.7 Esito del Benchmark Indipendente su `h_maxima` e Ripristino del Default (v4.2) — 19 agosto 2026

### 7.7.1 Risultati

Il benchmark di Sezione 7.6 punto 1 è stato eseguito su GPU (Colab) sulle **stesse 10 patch di validazione** usate in Sezione 3.2, con la medesima Ground Truth Cellpose v4.x ($d = 22.0$ px). Dati grezzi: [`data/fase2_segmentation/colab_benchmark_v3_hmaxima_vs_legacy.csv`](file:///c:/Users/Master/Desktop/testNuovoTesi/data/fase2_segmentation/colab_benchmark_v3_hmaxima_vs_legacy.csv).

| Metrica | WS `h_maxima` (v4.1) | WS `relative_threshold` (legacy) | Δ relativo |
| :--- | :---: | :---: | :---: |
| **Dice (pixel)** | $0.4973 \pm 0.1795$ | **$0.6373 \pm 0.1150$** | **−22.0%** |
| **IoU (pixel)** | $0.3478 \pm 0.1574$ | **$0.4763 \pm 0.1140$** | **−27.0%** |
| **AJI (instance)** | $0.2173 \pm 0.0933$ | **$0.3097 \pm 0.0723$** | **−29.8%** |
| **F1 Detection @0.5** | $0.2943 \pm 0.1078$ | **$0.4101 \pm 0.0755$** | **−28.2%** |

**Breakdown per classe istologica:**

| Classe | Metodo | Dice | AJI | F1 Det. |
| :--- | :--- | :---: | :---: | :---: |
| FL ($n=5$) | `h_maxima` | $0.4569$ | $0.2131$ | $0.2859$ |
| FL ($n=5$) | `relative_threshold` | **$0.6413$** | **$0.3240$** | **$0.4419$** |
| REACTIVE ($n=5$) | `h_maxima` | $0.5377$ | $0.2215$ | $0.3026$ |
| REACTIVE ($n=5$) | `relative_threshold` | **$0.6332$** | **$0.2953$** | **$0.3782$** |

Il degrado è **presente in entrambe le classi** ma più marcato su FL (Dice −28.8%) che su REACTIVE (−15.1%), coerentemente con la maggiore densità di impaccamento nucleare del linfoma, dove la separazione dei marker è più critica.

**Analisi patch per patch:** su **9 patch su 10** `h_maxima` perde su *tutte e quattro* le metriche simultaneamente. L'unica eccezione è `REACTIVE_examples (244)`, dove `h_maxima` vince su tutte e quattro (Dice $0.6898$ vs $0.6512$; AJI $0.3379$ vs $0.3198$) — un singolo caso, insufficiente a controbilanciare il quadro generale.

### 7.7.2 Decisione: Ripristino di `relative_threshold` come Default (v4.2)

Applicando il criterio di decisione dichiarato in Sezione 7.5 (*"se il nuovo metodo è peggiore, va ri-tarato `h_maxima_px` o si mantiene `relative_threshold` come default"*), il default di `segment_nuclei_watershed()` è stato **riportato a `marker_method="relative_threshold"`**. Il metodo `h_maxima` resta disponibile come opzione esplicita, non attiva.

**Conseguenza operativa:** le maschere e i centroidi già prodotti (94.042 nuclei, `centroids_all.csv`) sono stati generati con `relative_threshold` e **restano validi** — non è necessaria alcuna ri-esecuzione della Fase 2, e la Fase 3 può procedere sui dati esistenti.

### 7.7.3 Perché la Taratura di `h=5` Ha Fallito — Lezione Metodologica per la Tesi

Il parametro `h=5` era stato scelto in Sezione 7.5 cercando il valore che riproducesse più fedelmente il **conteggio totale di nuclei** del metodo storico (rapporto $0.84\times$ su 40 patch). Il benchmark dimostra che quel criterio di taratura era **il proxy sbagliato**: far coincidere *quanti* nuclei vengono rilevati non garantisce che i marker cadano nella *posizione corretta* per generare istanze ben sovrapposte alla Ground Truth. Un metodo può produrre il numero giusto di oggetti nei posti sbagliati.

> **Dichiarazione per la Sezione Metodologia/Limitazioni della tesi:**
> *"La taratura di un iperparametro di segmentazione deve essere condotta ottimizzando direttamente la metrica di qualità d'interesse (AJI/Dice contro una Ground Truth indipendente), non un proxy di numerosità come il conteggio degli oggetti rilevati. Nel presente lavoro, una taratura basata sul conteggio ha prodotto un parametro che degradava la qualità di segmentazione del 22-30% pur mantenendo un conteggio di nuclei apparentemente confrontabile."*

### 7.7.4 Ciò che Resta Valido dell'Analisi v4.1

Il risultato negativo **non invalida** la diagnosi tecnica di Sezione 7.2-7.3, che resta corretta e riproducibile:

- La soglia globale-relativa di `relative_threshold` **è** vulnerabile ai blob di grandi dimensioni (collasso 0/10 nuclei per raggio $> \approx 87$ px, Test 1), e `h_maxima` degrada gradualmente dove il default collassa (Sezione 7.5).
- Ciò che il benchmark dimostra è che questo vantaggio di **robustezza a casi limite** non si traduce, con la taratura attuale, in **accuratezza media su dati reali**. Sono due assi di valutazione distinti, e su questo dataset — dove `distance.max()` non supera mai $\approx 13$ px e lo scenario di collasso non si verifica mai — è il secondo a essere determinante.

Questa distinzione è utile in tesi: la vulnerabilità documentata resta una **limitazione dichiarata** del metodo adottato, rilevante qualora il lavoro venisse esteso a WSI o a dataset con eterogeneità dimensionale maggiore, dove il trade-off potrebbe invertirsi.

### 7.7.5 Nota sulla Correzione dell'AJI (FIX A) e sulla Varianza della U-Net

Due osservazioni emerse dal confronto tra questo run e quello di Sezione 3.2, **sulle stesse 10 patch**:

1. **AJI del Watershed legacy: $0.3097$ (qui) vs $0.3255$ (Sezione 3.2).** Dice, IoU e F1 sono invece *identici alla quarta cifra decimale*, e il Watershed è deterministico. La discrepanza è quindi imputabile esclusivamente alla metrica: questo run include il **FIX A** (Sezione 6), che ha corretto l'accoppiamento non univoco nel calcolo dell'AJI. Il valore **$0.3097$ è quello corretto**; $0.3255$ era lievemente inflazionato dal bug. *Da usare $0.3097$ nella tesi.*
2. **Metriche U-Net: forte varianza run-to-run.** La rete viene ri-addestrata a ogni esecuzione (15 epoche, inizializzazione e split casuali). Sulle stesse patch si osservano Dice $0.5738$ (Sezione 3.2) e $0.5062$ (questo run); in un terzo run su patch diverse ([`segmentation_benchmark_v2_cellpose_gt.csv`](file:///c:/Users/Master/Desktop/testNuovoTesi/data/fase2_segmentation/segmentation_benchmark_v2_cellpose_gt.csv)) la U-Net risultava addirittura *superiore* al Watershed ($0.6896$ vs $0.6269$). **La conclusione "Watershed > U-Net" di Sezione 4 va quindi presentata con cautela**: è solida sulle metriche d'istanza (AJI e F1, dove il Watershed vince in tutti i run disponibili), ma non robusta sul solo Dice pixel-level. Per una dichiarazione statisticamente difendibile in tesi servirebbe una media su più seed di addestramento — azione consigliata prima della consegna.

---

## 7.8 Il Benchmark Misurava una Configurazione Diversa dal Dataset (v4.3) — 20 agosto 2026

### 7.8.1 Il Fatto

Lo script Colab ([`scratch/run_colab_benchmark.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/scratch/run_colab_benchmark.py)) invocava la segmentazione **senza passare parametri**, ereditando i default del modulo. Quando i run furono eseguiti quei default erano `min_distance=12, min_area_px=30` — che, come stabilito nella Sezione 5.1, **non sono i parametri con cui è stato costruito il dataset della tesi**.

La verifica è diretta e non richiede la Ground Truth: `segmentation_benchmark_v2_cellpose_gt.csv` registra `ws_n_pred`, il numero di nuclei trovati dal Watershed durante il benchmark. Ricalcolandolo sulle stesse patch:

| Configurazione | Corrispondenze esatte con `ws_n_pred` | Scarto medio |
|---|---|---|
| `min_distance=12, min_area_px=30` | **10 / 10** | 0.00 |
| `min_distance=7, min_area_px=15` (dataset) | 0 / 10 | 78.10 |

### 7.8.2 Perché Importa

| | Nuclei per patch |
|---|---|
| Ground Truth Cellpose | **149.2** |
| Dataset della tesi (parametri corretti) | **141.8** |
| Watershed *come benchmarkato* | **63.7** |

Il Watershed è stato valutato in una configurazione che rileva il 43% dei nuclei trovati da Cellpose, mentre quella effettivamente usata per costruire il dataset ne rileva il 95%. Il recall di detection registrato ($0.23$–$0.26$) era quindi limitato alla radice, e con esso l'F1. **Le metriche pubblicate non descrivono la segmentazione su cui poggia la Fase 3.**

Non se ne può dedurre in che direzione cambieranno i risultati: un conteggio più vicino alla GT alza il recall, ma marker più fitti possono sovra-segmentare e penalizzare l'AJI. Va misurato, non previsto.

### 7.8.3 Perché Va Rifatto per Intero e non Ri-scorato

La U-Net è addestrata sulle maschere Watershed dei 20 patch di train. Cambiando i parametri cambia il suo **target di addestramento**: un confronto fra una U-Net addestrata sui vecchi target e un Watershed nuovo non avrebbe significato. Inoltre la Ground Truth Cellpose non è mai stata salvata su disco — era un dizionario in memoria, perduto con la sessione Colab — quindi non esiste nulla da ri-scorare.

### 7.8.4 Procedura per la Riesecuzione

1. `python scratch/build_colab_bundle.py` — assembla `colab_benchmark.zip` con le 30 patch dello Step 2.2 e il modulo di segmentazione **corrente**. Assemblarlo da script elimina il rischio di caricare su Colab una versione diversa del modulo, che è precisamente l'origine di questo problema.
2. Su Colab: caricare l'archivio, `!pip install cellpose`, `!python run_colab_benchmark.py`.
3. Riportare nel repository **tre** artefatti: `colab_benchmark_results.csv`, `benchmark_metadata.json` (versioni, parametri, split) e la cartella `cellpose_gt_masks/`. Senza il terzo il run tornerebbe non verificabile.
4. Aggiornare `segmentation_metadata.json`, la Sezione 4 di questo report e l'avviso nel README.

La versione `v4` dello script salva la GT su disco, passa i parametri esplicitamente tramite `WS_PARAMS` e registra la propria provenienza. [`tests/test_colab_benchmark_script.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/tests/test_colab_benchmark_script.py) fa da guardia sulle tre proprietà, ispezionando il sorgente: lo script non è eseguibile in locale, ma il ripetersi dell'errore è rilevabile.

### 7.8.5 Lezione Metodologica

È la seconda volta che il progetto paga lo stesso schema: **un runner fuori dal repository che chiama la pipeline coi default impliciti**. La prima è costata la riproducibilità delle 600 maschere (Sezione 5.1), la seconda la validità del benchmark. Il difetto non è nei valori dei parametri ma nel non dichiararli: un default è una decisione che si può cambiare senza accorgersi di aver cambiato anche tutti i risultati che vi dipendevano.
