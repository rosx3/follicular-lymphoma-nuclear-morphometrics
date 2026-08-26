# Report Fase 1 — Preprocessing delle Immagini Istologiche H&E
### Tesi: Quantificazione Citomorfometrica e Spaziale per la Classificazione tra Linfoma Follicolare e Tessuto Reattivo
*Modulo: [`src/01_preprocessing.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/src/01_preprocessing.py) — Versione 3.0*  
*Aggiornato il 18 agosto 2026 — Audit globale del progetto*

---

## Obiettivo

Trasformare le 600 patch H&E grezze in due rappresentazioni standardizzate:
1. **Immagine RGB normalizzata** — per l'analisi di tessitura e colore dello stroma.
2. **Canale Ematossilina (H-channel)** — input ad alto contrasto per la segmentazione dei nuclei.

---

## Calibrazione Spaziale

| Parametro | Valore | Riferimento |
|-----------|--------|-------------|
| **Scanner** | Hamamatsu NanoZoomer S360 C13220-01 | Carreras et al., *Cancers* (2025) |
| **Esportazione patch** | NDP.view2, 200× (= obiettivo 20×), 150 dpi | Carreras et al., *Cancers* (2025) |
| **Scala Spaziale** | **1 px = 0.46 µm** | Dedotta — vedi sezione successiva |
| **Dimensione Patch** | 224 × 224 px = **103.04 × 103.04 µm** | Calcolo analitico |
| **Area per Patch** | **10.617,24 µm²** | Calcolo analitico |

Tutti i biomarcatori delle fasi successive (area nucleare, perimetro, distanze inter-nucleari) sono espressi in **unità fisiche reali (µm, µm²)** per garantire interpretabilità clinica immediata e confrontabilità con la letteratura istopatologica (*Iwamoto et al., 2024*).

La costante è definita in un unico punto del codice, [`src/calibration.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/src/calibration.py), da cui tutti i moduli la importano.

---

## Come è stata determinata la scala spaziale

> **Nota (19 agosto 2026).** Fino a questa revisione il progetto usava 1 px = 0.23 µm.
> Quel valore si è rivelato errato. Questa sezione ricostruisce come ce ne siamo accorti
> e come siamo arrivati a quello attuale, perché il ragionamento è parte del metodo e
> va riportato nella tesi.

### 1. Un controllo di routine che non tornava

Durante l'implementazione delle distanze inter-nucleari della Fase 3, i biomarcatori
sono stati confrontati con i valori attesi in letteratura. Con la calibrazione di
0.23 µm/px i nuclei risultavano avere un diametro medio di **2,5 µm**. Un nucleo di
linfocita ne misura 6–7.

### 2. Non era colpa della segmentazione

Il primo sospetto era un difetto del nostro Watershed. È stato quindi confrontato il
diametro medio contro la Ground Truth, prodotta in modo indipendente: **2,51 µm**,
praticamente identico al nostro (2,48 µm).

Se anche il riferimento indipendente misura nuclei della stessa dimensione anomala,
il problema non è nell'algoritmo di segmentazione: è nei dati o nella loro interpretazione.

### 3. Il conto che rendeva l'errore certo

A questo punto sono stati messi insieme tre numeri:

- **154 nuclei** per patch — misurati da noi
- **≈33 µm²** l'area di un nucleo linfoide — valore di letteratura
- **2.654 µm²** l'area del campo visivo — conseguenza della calibrazione di 0.23 µm/px

Moltiplicando: `154 × 33 = 5.148 µm²` di nuclei da collocare in `2.654 µm²` di spazio
disponibile, cioè il **194%**. I nuclei avrebbero dovuto occupare quasi il doppio dello
spazio esistente.

Non è un risultato improbabile: è impossibile. Almeno uno dei tre numeri doveva essere
sbagliato.

### 4. Quale dei tre

Il conteggio dei nuclei è una misura diretta. La dimensione dei nuclei è un dato
consolidato. Il terzo numero, l'area del campo visivo, **non è una misura**: è una
conseguenza della calibrazione, che a sua volta era un'assunzione.

A confermare la direzione del sospetto c'è un fatto: alcune grandezze **non dipendono
affatto dalla calibrazione**, perché sono rapporti fra conteggi di pixel. La frazione
di area della patch occupata dai nuclei è una di queste, e vale **31,3%** — un valore
del tutto normale per un tessuto.

I pixel, quindi, raccontano una storia coerente. È la conversione da pixel a micron
a essere sbagliata.

### 5. Cosa dice davvero l'articolo sorgente

I 0.23 µm/px derivavano dal ragionamento: *scanner NanoZoomer S360 + obiettivo 40×
→ 0.23 µm/px*. Corretto come specifica tecnica dello scanner, ma rileggendo
Carreras et al. (2025) emerge che **l'obiettivo 40× non è mai nominato**. Gli autori
scrivono di aver convertito le immagini in JPEG a **200× e 150 dpi** tramite NDP.view2.

In istopatologia l'ingrandimento si esprime come prodotto obiettivo × oculare, dove
l'oculare è convenzionalmente 10×. Quindi:

| Notazione dell'articolo | Obiettivo corrispondente | Risoluzione |
|---|---|---|
| 400× (usato per alcune figure) | 40× | 0.23 µm/px — nativa dello scanner |
| **200× (usato per le patch)** | **20×** | **0.46 µm/px** |

Le patch sono state esportate a metà risoluzione. Il valore che stavamo usando era
quello corretto per un ingrandimento che non è stato impiegato per generarle.

### 6. La verifica indipendente

Una spiegazione plausibile non basta. Serviva un controllo che non fosse già stato
usato per arrivare alla conclusione: la **densità nucleare per millimetro quadro**,
grandezza ben documentata per il tessuto linfoide (**10.000–20.000 nuclei/mm²**).

| Calibrazione ipotizzata | Densità risultante | Esito |
|---|---|---|
| 0.23 µm/px (precedente) | 58.019 /mm² | impossibile |
| **0.46 µm/px (adottata)** | **14.505 /mm²** | **in intervallo** |
| 0.847 µm/px (lettura alternativa) | 4.282 /mm² | troppo rado per un centro germinativo |

Solo un valore regge, ed è lo stesso a cui porta la lettura dell'articolo. Il diametro
è coerente ma meno stringente: 4,96 µm misurati che, corretti per la sotto-copertura del
Watershed (Dice 0,795 rispetto alla Ground Truth), portano a **≈5,6 µm**, appena sotto
l'intervallo atteso per un linfocita. La correzione è però approssimativa — il Dice non è
la frazione di area coperta e la segmentazione produce anche falsi positivi — quindi
l'evidenza che discrimina fra le calibrazioni candidate resta la densità nucleare.

### 7. Cosa resta incerto

Gli autori **non pubblicano una scala esplicita**, né nell'articolo né nel record
Zenodo, e i JPEG riportano solo una densità JFIF generica di 96 dpi scritta al momento
del ritaglio. Il valore di 0.46 µm/px è quindi una **deduzione ben sostenuta da due
verifiche convergenti**, non un dato letto dalla fonte. Nella tesi va presentato come
tale. Una conferma definitiva richiederebbe di contattare gli autori o di misurare una
struttura di dimensione nota.

**Impatto della revisione.** Trattandosi di un fattore di scala globale, la correzione
è lineare sulle lunghezze (×2) e quadratica sulle aree (×4). Non altera ordinamenti fra
patch, test di separabilità né prestazioni dei modelli: cambia l'interpretazione
clinica assoluta e la confrontabilità con le soglie dimensionali della letteratura.
Le grandezze adimensionali (circolarità, eccentricità, solidità, aspect ratio,
coefficienti di variazione, tessitura, frazioni di area) non sono toccate.

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
           CLAHE (clipLimit=2.0, tile 28×28 px = 12.9 µm) aumenta
           il contrasto locale della cromatina nucleare.
```

---

## Scelte Progettuali Motivate e Riferimenti Bibliografici

### 1. Normalizzazione di Macenko vs. Reinhard
* **Scelta:** Normalizzazione cromatica di Macenko in Densità Ottica (OD).
* **Riferimento:** **Macenko et al. (2009)** — *IEEE ISBI*, pp. 1107-1110.
* **Motivazione Scientifica:** A differenza dei metodi di allineamento statistico nello spazio colore LAB (Reinhard), l'algoritmo di Macenko converte l'immagine in Densità Ottica ($OD = -\log_{10}((I+1)/256)$) ed applica la Decomposizione ai Valori Singoli (**SVD**) per stimare i reali vettori spettrali di assorbimento dell'Ematossilina e dell'Eosina. Questo rispetta la fisica dell'interazione luce-colorante regolata dalla legge di Beer-Lambert (*Macenko et al., 2009*).

  > **Nota Implementativa (v3.0):** La formula OD utilizza il divisore $256$ anziché $255$ affinché un pixel di intensità massima ($I=255$) produca $OD = -\log_{10}(256/256) = 0$, eliminando il rischio di valori OD fisicamente impossibili (negativi). L'offset $+1$ al numeratore garantisce che $I=0$ (assorbimento totale) non produca $\log(0)$.

### 2. Filtro Bilaterale vs. Filtro Gaussiano
* **Scelta:** Filtro Bilaterale con kernel $d=9$, $\sigma_{\text{color}}=75$, $\sigma_{\text{space}}=75$.
* **Motivazione Scientifica:** Il filtro Gaussiano attua una sfocatura uniforme che ammorbidisce e fonde i contorni delle membrane di nuclei adiacenti a contatto, ostacolandone la separazione. Il filtro bilaterale combina una componente spaziale e una cromatica, attenuando il rumore nelle regioni uniformi dello stroma ma **preservando la nitidezza dei gradienti di bordo nucleare**, essenziale per gli algoritmi di segmentazione (*Schmidt et al., 2018*).

### 3. Deconvoluzione Cromatica di Ruifrok & CLAHE Adattivo
* **Scelta:** Deconvoluzione Ruifrok & Johnston seguita da CLAHE su griglia 8×8.
* **Riferimenti:** **Ruifrok & Johnston (2001)** — *Anal. Quant. Cytol. Histol.* 23(4):291-9.
* **Motivazione Scientifica:** La matrice di deconvoluzione Ruifrok disaccoppia matematicamente il segnale dell'Ematossilina (nuclei) da quello dell'Eosina (citoplasma). Il CLAHE adattivo applicato su tile di 28×28 px (~12.9 × 12.9 µm²) opera a una scala spaziale confrontabile con quella di un nucleo linfocitario (5–10 µm) e del suo immediato intorno, aumentando il contrasto della cromatina nucleare senza amplificare il rumore stromatico globale (*Sung et al., 2024*).

---

## Fix Applicati dopo Audit (v1 → v2)

| Fix | Descrizione | Motivazione Scientifica |
|-----|-------------|-------------------------|
| 🔴 **FIX 1** | Allineamento stadi salvataggio (`denoised_rgb` per entrambi) | Garantisce coerenza tra le maschere nucleari e l'immagine RGB di overlay |
| 🔴 **FIX 2** | Reference image selezionata da mediana del dataset (`REACTIVE_examples (133).jpg`) | Evita distorsioni cromatiche derivanti dall'uso arbitrario della prima immagine alfabetica (distanza dalla mediana ridotta da 10.06 a 1.08) |
| 🟡 **FIX 3** | Refactoring SVD in `_estimate_HE_vectors()` | Principi di ingegneria del software DRY (Don't Repeat Yourself) |
| 🟡 **FIX 4** | Salvataggio `preprocessing_metadata.json` | Principi FAIR e riproducibilità scientifica della ricerca |

---

## Fix Applicati dopo Audit v2 → v3 (18 agosto 2026)

| Fix | File | Riga | Descrizione | Impatto |
|-----|------|------|-------------|---------|
| 🔴 **FIX 5** | `src/01_preprocessing.py` | ~293 | **Correzione formula OD:** `/ 255.0` → `/ 256.0` | Elimina il rischio di valori OD fisicamente negativi per pixel a intensità massima ($I=255$). Con $/ 255.0$, il valore $(255+1)/255 = 1.0039$ produceva $OD = -0.0017$, una violazione della legge di Beer-Lambert. Con $/ 256.0$, il valore $(255+1)/256 = 1.0$ produce $OD = 0.0$, fisicamente corretto (assorbimento nullo). |
| 🟢 **FIX 6** | `src/run_pipeline.py` | (nuovo file) | **Creato script di orchestrazione** `run_pipeline.py` come entry point principale | Gli script `01_preprocessing.py` e `02_segmentation.py` contenevano solo self-test interni. Il README dichiarava di eseguirli per processare il dataset, ma questo non accadeva. Il nuovo `run_pipeline.py` itera sulle 600 immagini reali con barre di progresso e gestione degli errori. |
| 🟢 **FIX 7** | `requirements.txt` | (nuovo file) | **Creato `requirements.txt`** con versioni pinned di tutte le dipendenze Python | Riproducibilità dell'ambiente: `numpy`, `scipy`, `scikit-image`, `opencv-python`, `torch`, `torchvision`, `xgboost`, `shap`. |
| 🟢 **FIX 8** | `.gitignore` | — | **Aggiornato `.gitignore`** con regole per `*.pth`, `venv/`, `.ipynb_checkpoints/`, `*.zip` | I pesi PyTorch da 93 MB erano stati caricati su GitHub con un avviso di dimensione. La regola `*.pth` previene push futuri di file pesanti. |

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
