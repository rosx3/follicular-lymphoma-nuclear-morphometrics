# Fase 4 — Classificazione Tabulare e Spiegabilità Clinica

*Spec di progetto — 20 agosto 2026*
*Modulo: `src/04_classification.py` — input: `data/fase3_features/features_patches_master.csv`*

---

## 1. Obiettivo

Distinguere **linfoma follicolare** da **tessuto linfoide reattivo** a partire dai 47
biomarcatori citomorfometrici, spaziali e di tessitura prodotti dalla Fase 3, e
spiegare **quali biomarcatori guidano la decisione e in che direzione**.

La spiegabilità non è un complemento: è la ragione per cui la tesi adotta un
approccio white-box invece di una CNN end-to-end. Un modello che classificasse
bene senza dire *perché* mancherebbe l'obiettivo dichiarato.

**Criterio di successo.** Non una soglia di accuratezza, ma tre cose:

1. una stima **onesta** delle prestazioni, con il leakage misurato invece che ignorato;
2. una gerarchia di biomarcatori **coerente** con i test univariati della Fase 3
   (`lbp_entropy` primo, famiglia solidità/circolarità irrilevante) — una forte
   incoerenza sarebbe il segnale di un errore, non di una scoperta;
3. risultati **riproducibili**: stesso comando, stessi numeri.

---

## 2. Input e vincoli

| | |
|---|---|
| Righe | 600 patch (300 FL, 300 REACTIVE) — perfettamente bilanciate |
| Colonne | 47 biomarcatori + `image_name`, `category`, `target` |
| Valori mancanti | **nessuno** (Fase 3, §3.1): non serve alcuna imputazione |
| Scala | eterogenea (µm, µm², conteggi, adimensionali) |
| Etichette di caso/paziente | **assenti** — vedi D1 |

Vincolo di riproducibilità del progetto: seed fissi, versioni pinnate in
`requirements.txt` (`scikit-learn 1.8.0`, `xgboost 3.2.0`, `shap 0.51.0`).

---

## 3. Decisioni di progetto

### D1 — Doppia validazione, per misurare il leakage invece di subirlo

**Problema.** Le 600 patch non provengono da 600 pazienti. La serie di origine
(Carreras et al. 2025) è di **221 casi** — 177 FL e 44 reattivi — da cui sono
state estratte ~1,5 milioni di patch: più patch per caso, quindi, e le nostre 600
ne sono un campione. Il dataset pubblicato su Zenodo
(`Examples_of_images_v20250620.zip`) è **piatto**: due cartelle, 300 file
ciascuna, nessun identificativo di caso. L'informazione non è recuperabile.

Con uno split casuale, patch dello stesso vetrino finiscono in addestramento e in
test. Il modello può allora imparare la firma del vetrino — intensità della
colorazione, resa dello scanner, spessore della sezione — invece della biologia.
Le feature più esposte sono proprio `hchannel_mean`, `glcm_contrast`,
`glcm_homogeneity` e `lbp_entropy`, che misurano intensità e tessitura del
segnale. Il punteggio risultante risponderebbe a una domanda più facile e
clinicamente inutile: *«riconosco altre patch di pazienti che ho già visto»*.

**Verifica empirica.** Il rischio non è teorico su questi dati. Ordinando le
patch per indice, la distanza media nello spazio dei biomarcatori fra coppie
**adiacenti** vale $0.615\times$ (FL) e $0.691\times$ (REACTIVE) quella fra
coppie qualsiasi, con correlazione indice↔somiglianza a $p \approx 0$ in entrambe
le classi. L'ordine di numerazione conserva quindi una struttura a blocchi
compatibile con l'esportazione caso per caso.

**Decisione.** Ogni modello viene valutato **due volte, con gli stessi dati e lo
stesso codice**, cambiando solo lo splitter:

- **Validazione A — `StratifiedKFold(5)`**: split casuale. Stima *ottimistica*,
  confrontabile con la letteratura patch-level.
- **Validazione B — `GroupKFold(5)` su blocchi contigui di indice**: le patch
  vicine restano dalla stessa parte. Stima *conservativa*.

**Ciò che si pubblica è la forbice fra le due.** Se è stretta, il leakage non
incideva e lo si dichiara con un numero a supporto. Se è larga, lo si è scoperto
prima della discussione invece di pubblicare il valore gonfiato.

I blocchi **non pretendono di identificare i pazienti**: sono una sonda. Per non
far dipendere la conclusione da un parametro arbitrario, la validazione B viene
ripetuta con blocchi da **5, 10, 20 e 30** patch e se ne riporta l'andamento.
Se la metrica degrada al crescere del blocco, la dipendenza dal vicinato è reale.

Il risultato principale si riporta con **blocchi da 10**, scelto *prima* di
vedere le metriche e non ottimizzato su di esse: è dell'ordine di grandezza
suggerito dalla serie di origine (300 patch reattive per 44 casi, ≈ 7 patch per
caso) con un margine di prudenza. Gli altri tre valori accompagnano sempre il
principale come analisi di sensibilità: se la conclusione cambiasse al variare
del blocco, è quel fatto a dover essere riportato, non il valore più favorevole.

### D2 — Tre modelli, di cui uno di riferimento

- **Regressione logistica** (con standardizzazione): il riferimento minimo. Se un
  modello lineare regge il confronto con gli alberi, la complessità non serve —
  ed è un risultato da riportare, non una sconfitta.
- **Random Forest** e **XGBoost**: quelli dichiarati dal README, adatti a 600
  righe per ~33 colonne e invarianti a riscalature (Fase 3, §5).

La standardizzazione vive **dentro** la `Pipeline` di scikit-learn, mai applicata
prima dello split: adattare lo scaler sull'intero dataset è leakage, della stessa
famiglia di D1.

### D3 — Riduzione delle ridondanze prima di SHAP

**Problema.** `n_nuclei` e `nuclear_density_per_1000um2` sono la stessa
grandezza in unità diverse (Spearman $= 1{,}0000$); la Fase 3 (§3.3) individua **9
coppie con $|\rho| > 0{,}95$**. Gli alberi tollerano la collinearità, ma SHAP no:
fra due variabili quasi identiche il merito viene diviso arbitrariamente e
**entrambe appaiono meno importanti di quanto siano**. Si comprometterebbe
proprio l'obiettivo interpretativo.

**Decisione.** Clustering gerarchico sulla matrice $1 - |\rho_{Spearman}|$,
taglio a $|\rho| > 0{,}90$ (→ 33 gruppi), e di ogni gruppo si tiene **una sola**
variabile, scelta per **leggibilità clinica**.

La scelta non è ad hoc gruppo per gruppo: il modulo definisce una costante
`READABILITY_ORDER`, una classifica esplicita e commentata delle famiglie di
biomarcatori, e il rappresentante di ogni gruppo è il membro meglio piazzato in
quella classifica. Criterio unico, deterministico, ispezionabile.

La preferenza è per la grandezza **direttamente misurabile e nominabile da un
patologo**: `n_nuclei` prima della densità derivata, `area_um2_mean` prima
dell'asse minore, `circularity_mean` prima della sua asimmetria.

*Perché non il criterio statistico.* Tenere di ogni gruppo la variabile con
l'effect size maggiore sarebbe più oggettivo, ma su questi dati sceglie
sistematicamente la variabile meno comprensibile: terrebbe `knn3_dist_mean_um`
scartando `n_nuclei`, e `circularity_skew` scartando `circularity_mean`. Poiché
l'output della Fase 4 è una spiegazione clinica, la leggibilità prevale — e la
scelta viene documentata gruppo per gruppo in una tabella del report, così che
il lettore possa dissentire con cognizione.

Le variabili scartate **non spariscono**: la tabella riporta, per ognuna, il
gruppo di appartenenza e il rappresentante che la sostituisce.

### D4 — Taratura annidata degli iperparametri

Gli iperparametri si scelgono con una **cross-validation interna** dentro ogni
piega di addestramento (`GridSearchCV` annidato), mai sull'intero dataset.
Sceglierli guardando anche i dati di test gonfierebbe le metriche esattamente
come il leakage di D1: è lo stesso errore, applicato ai parametri invece che alle
righe.

Griglie deliberatamente piccole (poche decine di combinazioni per modello): con
600 righe una ricerca ampia sovradatta la ricerca stessa, e il costo
computazionale della doppia validazione si moltiplica.

### D5 — Metriche

**Principale: AUC-ROC**, indipendente dalla soglia e adatta a classi bilanciate.
Accanto: **accuratezza bilanciata**, **sensibilità** e **specificità**, che sono
i termini con cui un clinico legge un test diagnostico.

Non si applica alcun bilanciamento di classe: 300 contro 300.

La soglia decisionale resta a $0{,}5$ e non viene ottimizzata. Sceglierla sui
dati di test sarebbe un'altra forma dello stesso errore; la lettura per soglie
diverse è già contenuta nella curva ROC.

### D6 — Spiegazione con SHAP

Sul modello migliore secondo la validazione **B** (la conservativa):

- **importanza globale**: media dei valori assoluti, con direzione dell'effetto;
- **coerenza con la Fase 3**: confronto fra la gerarchia SHAP e quella degli
  effect size univariati, dichiarando dove divergono. Una divergenza è
  interessante — indica un effetto che emerge solo in presenza di altre variabili
  — ma va distinta da un errore;
- **due casi esemplari**: una patch classificata correttamente con alta
  confidenza e una sbagliata, per mostrare come si legge una spiegazione locale.

Per gli alberi si usa `TreeExplainer` (esatto); per la logistica i coefficienti
standardizzati, che sono già la spiegazione.

---

## 4. Architettura del modulo

Funzioni piccole e testabili singolarmente; `main()` si limita a orchestrare.

| Funzione | Responsabilità | Dipendenze |
|---|---|---|
| `load_feature_matrix(csv_path)` | legge il CSV, restituisce `X`, `y`, nomi delle feature e indici di patch | pandas |
| `contiguous_blocks(image_names, block_size)` | dagli `image_name` ricava l'indice numerico e assegna a ogni patch un id di blocco, **per categoria** | — |
| `redundancy_groups(X, threshold)` | clustering gerarchico su $1-\|\rho\|$, restituisce i gruppi | scipy |
| `select_representatives(groups, readability_order)` | un rappresentante per gruppo secondo la classifica di leggibilità | — |
| `build_models(seed)` | i tre stimatori con le rispettive griglie, ciascuno in una `Pipeline` | sklearn, xgboost |
| `evaluate(X, y, groups, splitter, models, seed)` | CV annidata, restituisce le metriche per modello e per piega | sklearn |
| `explain_with_shap(model, X, feature_names)` | valori SHAP e importanza globale | shap |
| `main()` | esegue A e B, la sensibilità al blocco, salva output e figure | — |

**Flusso dei dati.**

```
features_patches_master.csv
        │
        ▼  load_feature_matrix
   X (600×47), y, image_name
        │
        ▼  redundancy_groups + select_representatives
   X (600×~33)  ──────────────┬───────────────────────┐
        │                     │                       │
        ▼ StratifiedKFold(5)  ▼ GroupKFold(5)         ▼ blocchi 5/10/20/30
   Validazione A          Validazione B          sensibilità
        └──────────┬──────────┘
                   ▼
            forbice + SHAP sul migliore di B
```

---

## 5. Output prodotti

| Percorso | Contenuto |
|---|---|
| `data/fase4_classification/metrics_by_model.csv` | metriche per modello × validazione × piega |
| `data/fase4_classification/block_size_sensitivity.csv` | metriche al variare della dimensione del blocco |
| `data/fase4_classification/feature_reduction.csv` | 47 righe: feature, gruppo, tenuta/scartata, rappresentante |
| `data/fase4_classification/shap_importance.csv` | importanza globale e direzione |
| `data/fase4_classification/best_model.joblib` | modello riaddestrato su tutti i dati |
| `data/fase4_classification/classification_metadata.json` | seed, versioni, griglie, parametri scelti |
| `img/fase4/roc_curves.png` | ROC dei tre modelli, validazioni A e B |
| `img/fase4/validation_gap.png` | la forbice A vs B, per modello |
| `img/fase4/shap_summary.png` | beeswarm SHAP del modello migliore |
| `img/fase4/shap_vs_univariate.png` | gerarchia SHAP contro effect size della Fase 3 |
| `reports/fase4_report.md` | metodo, risultati, limitazioni |

---

## 6. Analisi dei risultati

Calcolare le metriche non è la fine della Fase 4: **la fase si considera conclusa
quando i numeri sono stati letti**. Questa sezione definisce cosa
`reports/fase4_report.md` deve rispondere. Non è un elenco di buone intenzioni:
sono le domande che il modulo produce i dati per risolvere, e ognuna ha un
artefatto corrispondente in §5.

**A. Quanto valeva il leakage.** Per ciascun modello, la differenza fra
validazione A e B su AUC-ROC e accuratezza bilanciata, in valore assoluto e
relativo. È il risultato metodologico principale della fase.
→ `metrics_by_model.csv`, `img/fase4/validation_gap.png`

**B. La forbice dipende dal parametro?** Andamento della metrica al crescere del
blocco (5→30). Un degrado monotono conferma la dipendenza dal vicinato; un
andamento piatto indica che i blocchi non catturavano struttura, e allora la
stima A e la B vanno considerate equivalenti — conclusione altrettanto valida,
purché dichiarata.
→ `block_size_sensitivity.csv`

**C. Quale modello, e la differenza conta?** Confronto dei tre sulla validazione
B. La differenza fra due modelli va giudicata **rispetto alla variabilità fra le
pieghe**, non in assoluto: se le distribuzioni per piega si sovrappongono, la
risposta corretta è «indistinguibili», come è già accaduto per Watershed e U-Net
nella Fase 2. In particolare va detto **se la regressione logistica regge il
confronto**: in tal caso la complessità degli alberi non è giustificata su questi
dati, ed è un risultato da riportare.
→ `metrics_by_model.csv`, `img/fase4/roc_curves.png`

**D. I biomarcatori che decidono sono gli stessi della Fase 3?** Confronto fra la
gerarchia SHAP e quella degli effect size univariati. Ci si attende `lbp_entropy`
in alto e la famiglia solidità/circolarità in basso (Fase 3, §3.3). Una forte
incoerenza va trattata come **sospetto di errore** prima che come scoperta, e
indagata. Le divergenze legittime — una variabile debole da sola ma utile in
combinazione — vanno distinte e spiegate.
→ `shap_importance.csv`, `img/fase4/shap_vs_univariate.png`

**E. In che direzione?** Per i primi biomarcatori: valori alti verso FL o verso
REACTIVE, e coerenza col quadro clinico della Fase 3 (nel linfoma follicolare
nuclei più piccoli e allungati, packing meno fitto, cromatina più uniforme). È il
punto in cui la tesi passa da «il modello funziona» a «il modello dice qualcosa
di biologicamente sensato».
→ `img/fase4/shap_summary.png`

**F. Quanto sono incerti questi numeri?** Intervalli di confidenza sulle metriche
principali. Con poche unità indipendenti nella validazione B saranno ampi: vanno
riportati accanto a ogni media, mai la media da sola.
→ `metrics_by_model.csv`

**G. Dove sbaglia.** Le patch classificate male sono concentrate in qualche
blocco o sparse? Una concentrazione suggerisce un effetto di vetrino — cioè
proprio ciò che D1 teme — e va detta.
→ `metrics_by_model.csv` (predizioni per piega)

Una domanda a cui il report **non** risponde è una domanda che va tolta da qui,
non lasciata cadere in silenzio.

---

## 7. Gestione degli errori

Coerente col principio del progetto: **un input mancante è un errore visibile,
non un risultato silenzioso**.

- CSV assente o privo di una colonna attesa → `FileNotFoundError` / `ValueError`
  con il nome della colonna mancante e il comando per rigenerarlo.
- Valori mancanti nella matrice → errore esplicito. La Fase 3 garantisce che non
  ce ne siano: se compaiono, qualcosa a monte è cambiato e va indagato, non
  imputato in silenzio.
- Un blocco che finirebbe a cavallo di due pieghe → errore. È la garanzia stessa
  della validazione B.
- `image_name` da cui non si ricava un indice numerico → errore: la validazione B
  dipende da quell'ordine.

---

## 8. Test

Nello stile delle guardie già presenti nel progetto (`test_calibration.py`,
`test_segmentation_reproducibility.py`): pochi test, ciascuno su una proprietà
che, se violata, invaliderebbe un numero della tesi.

1. **Nessuna sovrapposizione di gruppi fra addestramento e test** nella
   validazione B — è la proprietà che giustifica l'intera D1.
2. **I blocchi sono contigui e per categoria**: patch FL e REACTIVE non finiscono
   mai nello stesso blocco.
3. **La riduzione elimina esattamente un rappresentante per gruppo**, e
   l'insieme tenuto è un sottoinsieme di `PATCH_FEATURE_COLUMNS`.
4. **`n_nuclei` sopravvive e `nuclear_density_per_1000um2` no**: caso concreto e
   verificabile della regola di leggibilità.
5. **Lo scaler è dentro la Pipeline**: nessuna trasformazione adattata fuori dallo
   split (ispezione della struttura, come le guardie sui default).
6. **Riproducibilità**: due esecuzioni con lo stesso seed danno le stesse metriche.
7. **Contratto di output**: i CSV prodotti hanno le colonne dichiarate qui.
8. **La validazione B non è più ottimistica della A** su nessun modello: se lo
   fosse, l'inferenza dei blocchi non starebbe catturando nulla e la conclusione
   andrebbe riesaminata.

I test girano su un sottoinsieme del dataset dove possibile, per restare rapidi.

---

## 9. Limitazioni da dichiarare nella tesi

1. **Assenza di etichette di caso.** Il partizionamento per paziente, che
   Carreras et al. eseguono esplicitamente (*«hybrid partitioning… using a
   patient-level independent validation set»*), qui **non è possibile**: il
   dataset pubblicato non contiene identificativi di caso. La validazione B è
   un'approssimazione basata sull'ordine di numerazione, non una garanzia.

2. **Confronto con Carreras et al. non diretto.** Gli autori riportano
   **99,8%** di accuratezza patch-level. Il numero **non è confrontabile** con
   quello di questa tesi, per tre ragioni che vanno dette insieme al confronto:
   (a) è ottenuto su ~1,5 milioni di patch contro le 600 qui disponibili;
   (b) con partizionamento a livello di paziente, che qui non è realizzabile;
   (c) da una CNN end-to-end sui pixel, mentre qui si classificano 33 biomarcatori
   interpretabili. Citarlo come termine di paragone senza queste tre precisazioni
   darebbe l'impressione di un divario di prestazioni, quando è in larga parte un
   divario di dati e di compito.

3. **Numerosità.** 600 patch, e nella validazione B il numero effettivo di unità
   indipendenti è dell'ordine delle decine. Gli intervalli di confidenza saranno
   ampi e vanno riportati, non nascosti dietro la media.

4. **A monte.** La Fase 4 eredita i limiti della Fase 2 già dichiarati: la
   segmentazione recupera l'85% dei nuclei rilevati da Cellpose, quindi i
   biomarcatori sono calcolati su una popolazione nucleare incompleta ma
   sistematica.

---

## 10. Fuori scope

- **Reti neurali sui pixel.** Non erano previste: il README definisce la tesi
  *in opposizione* agli approcci black-box end-to-end. L'unica rete del progetto
  resta la U-Net della Fase 2, che segmenta.
- **Integrazione nella GUI.** La tab «Analizza immagine» continuerà a dichiarare
  che non fornisce diagnosi. Collegare il modello è un passo successivo, da
  decidere quando i risultati saranno noti.
- **Selezione di feature guidata dalle prestazioni.** Oltre alla riduzione delle
  ridondanze di D3 non si fa selezione: sceglierle massimizzando la metrica
  reintrodurrebbe leakage.
