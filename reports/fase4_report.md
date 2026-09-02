# Report Finale — Fase 4: Classificazione Tabulare e Spiegabilità Clinica

### Tesi: Quantificazione Citomorfometrica e Spaziale per la Classificazione tra Linfoma Follicolare e Tessuto Reattivo

*Modulo: [`src/04_classification.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/src/04_classification.py)*
*Spec: [`docs/superpowers/specs/2026-08-20-fase4-classificazione-design.md`](file:///c:/Users/Master/Desktop/testNuovoTesi/docs/superpowers/specs/2026-08-20-fase4-classificazione-design.md)*
*Eseguito il 20 agosto 2026 — seed 42, artefatti in `data/fase4_classification/`*

---

## 1. Sintesi Esecutiva

Le 600 patch sono classificate a partire da **33 biomarcatori** (ridotti dai 47 della Fase 3 eliminando le ridondanze), con tre modelli a confronto e **due validazioni**: una casuale e una a blocchi, per misurare quanto del punteggio dipenda dal fatto che patch dello stesso caso finiscano da entrambe le parti dello split.

| | Risultato |
|---|---|
| **Miglior modello** | XGBoost |
| **Stima conservativa (validazione a blocchi)** | **AUC-ROC $0.9401$**, accuratezza bilanciata $0.8617$ |
| Stima ottimistica (split casuale) | AUC-ROC $0.9648$ |
| **Forbice fra le due** | $+0.0247$ di AUC |
| **Biomarcatore dominante** | `lbp_entropy` (importanza SHAP $3.15$, il doppio del secondo) |

**Il numero da portare in tesi è $0.9401$**, non $0.9648$: il secondo include il vantaggio di aver visto in addestramento altre patch degli stessi vetrini.

Due analisi successive precisano *da dove* viene quel risultato:

| | Esito |
|---|---|
| **Contributo per famiglia** (§4.4) | Cinque biomarcatori di tessitura e intensità arrivano a $0.944$; i 28 morfometrici e spaziali da soli si fermano a $0.857$. **La tessitura cromatinica porta il grosso del segnale**, come indicazione coerente su tre modelli. |
| **Robustezza alla colorazione** (§5) | Perturbando artificialmente la colorazione, a $\sigma=0.2$ il modello perde mezzo punto di AUC e cambia idea su 2 patch su 100. **Quelle feature leggono la cromatina, non il vetrino.** |

---

## 2. Dati e Riduzione delle Ridondanze

Input: `features_patches_master.csv`, 600 patch × 47 biomarcatori, classi perfettamente bilanciate (300/300), nessun valore mancante.

Il clustering gerarchico su $1 - |\rho_{Spearman}|$ con taglio a $|\rho| > 0{,}90$ produce **33 gruppi**. Di ciascuno si tiene una sola variabile, scelta per **leggibilità clinica** (decisione D3): la grandezza che un patologo nomina e misura vince sulla sua derivata o sui suoi momenti di ordine superiore.

Le 14 variabili scartate e chi le rappresenta:

| Scartata | Rappresentata da |
|---|---|
| `nuclear_density_per_1000um2` | `n_nuclei` |
| `knn3_dist_mean_um` | `n_nuclei` |
| `area_top10_short_axis_um`, `area_um2_std` | `area_top10_mean_um2` |
| `minor_axis_um_mean` | `area_um2_mean` |
| `major_axis_um_mean` | `perimeter_um_mean` |
| `aspect_ratio_mean` | `eccentricity_mean` |
| `aspect_ratio_cv` | `aspect_ratio_std` |
| `circularity_skew` | `circularity_mean` |
| `circularity_cv` | `circularity_std` |
| `eccentricity_cv` | `eccentricity_std` |
| `solidity_cv` | `solidity_std` |
| `knn3_dist_std_um` | `knn1_dist_std_um` |
| `glcm_energy` | `glcm_homogeneity` |

Tabella completa in `feature_reduction.csv`. La riduzione serve alla spiegabilità, non alle prestazioni: fra due variabili quasi identiche SHAP divide il merito arbitrariamente e le fa apparire entrambe meno importanti di quanto sono.

---

## 3. Validazione e Risultati

### 3.0 Perché due validazioni, e perché blocchi contigui

La convalida incrociata presuppone che le unità usate per addestrare e quelle usate per valutare siano indipendenti fra loro (*Arlot & Celisse, 2010*). Quando non lo sono, la stima che ne esce è ottimistica. Qui l'ipotesi non regge. Le 600 patch provengono da una serie di 221 casi (*Carreras et al., 2025*), quindi più patch condividono lo stesso vetrino. Il dataset pubblicato è però piatto e non contiene identificativi di caso, per cui il raggruppamento corretto non è ricostruibile.

**Il rischio è documentato nel dominio.** Bussola et al. (*2021*) hanno misurato l'effetto su 3 dataset istopatologici, 374 soggetti, 556 vetrini e oltre 27.000 tile, in 4 compiti di classificazione. Quando tile dello stesso soggetto compaiono sia in addestramento sia in validazione, i punteggi risultano gonfiati fino al $41\%$. Il limite del lavoro, rispetto al nostro caso, è che la partizione corretta che propone richiede le etichette di soggetto: sono proprio quelle che qui mancano.

La causa del fenomeno è a sua volta documentata. Howard et al. (*2021*) mostrano su oltre 3.000 pazienti del TCGA che la firma del centro di provenienza è riconoscibile dalle immagini istologiche, che sopravvive alla normalizzazione cromatica e all'augmentation, e che produce accuratezze distorte nella predizione di sopravvivenza, mutazioni e stadio. Il limite è che il lavoro riguarda reti convoluzionali su immagini, non biomarcatori tabulari: la trasferibilità al nostro caso è plausibile ma non dimostrata dagli autori.

**Il metodo adottato ha un nome.** Mancando le etichette di caso, si usa come sostituto l'ordine di numerazione dei file, sul presupposto che l'esportazione sia avvenuta caso per caso. Non è un espediente costruito per l'occasione. Burman, Chow e Nolan (*1994*) estendono la convalida incrociata alle sequenze stazionarie con la **h-block cross-validation**: dall'insieme di addestramento si rimuovono le $h$ osservazioni che precedono e le $h$ che seguono l'osservazione di test. L'ipotesi richiesta è che la dipendenza decada con la distanza lungo l'ordinamento. Roberts et al. (*2017*) generalizzano il criterio a strutture non temporali, siano esse spaziali, gerarchiche o filogenetiche, e fissano il principio per dimensionare il blocco: deve superare la portata della dipendenza.

**L'ipotesi richiesta dalla teoria è stata verificata, non assunta.** Dentro ciascuna classe, la distanza media nello spazio dei 47 biomarcatori standardizzati vale:

| Classe | Coppie adiacenti | Coppie qualsiasi | Rapporto |
|---|:---:|:---:|:---:|
| Linfoma follicolare | $5.6689$ | $9.2153$ | $\mathbf{0.6152}$ |
| Tessuto reattivo | $6.3894$ | $9.2529$ | $\mathbf{0.6905}$ |

Test di permutazione su 2000 estrazioni: $p = 0.0005$ in entrambe le classi, il minimo rappresentabile con quel numero di estrazioni. L'ordine di numerazione conserva quindi struttura. Verifica riproducibile con `python src/block_structure.py`, che riscrive `block_structure_evidence.csv`.

**Limite del nostro adattamento.** h-block rimuove un intorno attorno a ogni punto di test. La validazione B usa invece `GroupKFold` su blocchi interi: una patch che cade sul bordo di un blocco ha ancora i propri vicini immediati nell'insieme di addestramento, perché appartengono al blocco successivo. Con blocchi da 10 la cosa riguarda 2 patch su 10, e rende il nostro schema leggermente meno severo di un h-block vero. Il rimedio noto è il cuscinetto introdotto dalla variante **hv-block** (*Racine, 2000*), che dimostra asintoticamente ottimale là dove h-block non lo è. Non è stato applicato in questo lavoro.

Va infine ribadito ciò che i blocchi non sono. Non identificano i pazienti. Sono una sonda per misurare quanto il punteggio dipenda dal vicinato, e la stima che producono va letta come limite inferiore.

### 3.1 A. Quanto valeva il leakage

| Modello | A — split casuale | B — blocchi da 10 | Forbice |
|---|:---:|:---:|:---:|
| Regressione logistica | $0.9181 \pm 0.0113$ | $0.8992 \pm 0.0730$ | $+0.0189$ |
| Random Forest | $0.9602 \pm 0.0065$ | $0.9361 \pm 0.0437$ | $+0.0240$ |
| **XGBoost** | $0.9648 \pm 0.0116$ | $\mathbf{0.9401 \pm 0.0350}$ | $+0.0247$ |

*(AUC-ROC, media ± deviazione standard sulle 5 pieghe. L'accuratezza bilanciata dà la stessa storia: forbice $+0.022$, $+0.028$, $+0.030$.)*

**La forbice è stretta**: circa due punti di AUC, coerente su tutti e tre i modelli. Il leakage era presente ma pesava poco, e la stima onesta resta alta. È il risultato migliore fra i due possibili — ma va sottolineato che è un risultato *misurato*, non assunto: prima di questa fase non c'era modo di sapere se valesse due punti o venti.

### 3.2 B. La forbice dipende dalla dimensione del blocco?

| Modello | blocchi 5 | 10 | 20 | 30 |
|---|:---:|:---:|:---:|:---:|
| Regressione logistica | $0.9107$ | $0.8992$ | $0.8894$ | $0.8570$ |
| Random Forest | $0.9498$ | $0.9361$ | $0.9347$ | $0.9150$ |
| XGBoost | $0.9603$ | $0.9401$ | $0.9381$ | $0.9349$ |

**Il degrado è monotono per tutti e tre**: più patch vicine si tengono unite, più la stima scende. La dipendenza dal vicinato è quindi reale, non un artefatto della scelta del parametro — se i blocchi non catturassero struttura, la curva sarebbe piatta.

Da notare la differenza di robustezza: XGBoost perde $0.025$ passando da blocchi di 5 a blocchi di 30, la regressione logistica $0.054$ — più del doppio. **Il modello lineare si appoggiava di più al segnale che sparisce isolando i blocchi.**

### 3.3 C. Quale modello, e la differenza conta?

Le AUC per piega nella validazione B:

| Piega | Logistica | Random Forest | XGBoost |
|---|:---:|:---:|:---:|
| 0 | $0.9372$ | $0.9746$ | $0.9669$ |
| 1 | $0.9819$ | $0.9736$ | $0.9803$ |
| 2 | $0.9536$ | $0.9549$ | $0.9564$ |
| 3 | $0.8075$ | $0.9192$ | $0.8939$ |
| 4 | $0.8156$ | $0.8585$ | $0.9028$ |

Confronti appaiati (Wilcoxon, $n=5$):

- **XGBoost vs Random Forest**: $+0.0039$, $p = 1{,}000$ → **indistinguibili**.
- XGBoost vs logistica: $+0.0409$, $p = 0{,}125$ → non significativo con 5 pieghe.
- Random Forest vs logistica: $+0.0370$, $p = 0{,}188$.

**Conclusione onesta: i due modelli ad albero sono equivalenti fra loro, e il loro vantaggio sulla regressione logistica non è statisticamente dimostrato** con questa numerosità, pur essendo consistente in direzione (4 pieghe su 5) e di entità non trascurabile ($+0.04$).

Il fatto che un modello lineare arrivi a $0.899$ dice che **gran parte del segnale discriminante è lineare**. Gli alberi aggiungono qualcosa, ma la struttura del problema non è prevalentemente non lineare. Il riferimento minimo previsto da D2 si è rivelato informativo proprio per questo.

### 3.4 F. Pannello di metriche e intervalli di confidenza

Una sola metrica non basta, per due ragioni distinte. La prima è di merito: l'AUC misura l'ordinamento, non le decisioni, e una diagnosi differenziale si fa con decisioni. La seconda è di confronto: i lavori di riferimento non riportano la stessa grandezza, e senza un pannello non esiste una voce su cui il confronto sia diretto (§3.6).

Media ± $1{,}96 \times$ errore standard sulle 5 pieghe, validazione a blocchi, soglia $0{,}5$:

| Metrica | XGBoost | Random Forest | Logistica |
|---|:---:|:---:|:---:|
| AUC-ROC | $\mathbf{0.9401}$ $[0.9057,\ 0.9744]$ | $0.9361$ $[0.8933,\ 0.9790]$ | $0.8992$ $[0.8276,\ 0.9707]$ |
| Accuratezza | $\mathbf{0.8617}$ $[0.8156,\ 0.9077]$ | $0.8433$ $[0.7903,\ 0.8964]$ | $0.8267$ $[0.7625,\ 0.8908]$ |
| Precisione | $0.8930$ $[0.8820,\ 0.9039]$ | $0.8631$ $[0.8486,\ 0.8775]$ | $0.8431$ $[0.8168,\ 0.8694]$ |
| Richiamo (sensibilità) | $0.8233$ $[0.7094,\ 0.9372]$ | $0.8167$ $[0.6893,\ 0.9440]$ | $0.8033$ $[0.6496,\ 0.9570]$ |
| Specificità | $0.9000$ $[0.8769,\ 0.9231]$ | $0.8700$ $[0.8419,\ 0.8981]$ | $0.8500$ $[0.8100,\ 0.8900]$ |
| F1 | $0.8519$ $[0.7932,\ 0.9106]$ | $0.8337$ $[0.7655,\ 0.9019]$ | $0.8148$ $[0.7314,\ 0.8982]$ |
| **False negative rate** | $\mathbf{0.1767}$ $[0.0628,\ 0.2906]$ | $0.1833$ $[0.0560,\ 0.3107]$ | $0.1967$ $[0.0430,\ 0.3504]$ |
| False positive rate | $0.1000$ $[0.0769,\ 0.1231]$ | $0.1300$ $[0.1019,\ 0.1581]$ | $0.1500$ $[0.1100,\ 0.1900]$ |

Nella validazione A, per confronto, XGBoost dà AUC $0.9648$ $[0.9534,\ 0.9762]$ e accuratezza $0.8917$ $[0.8639,\ 0.9195]$. Gli intervalli della validazione B sono **tre volte più ampi** di quelli della A. Non è un difetto della validazione conservativa: è l'incertezza reale, che lo split casuale nascondeva contando come indipendenti patch che non lo sono. **Ogni valore va sempre citato col suo intervallo.**

#### Come leggere questo pannello

**L'accuratezza bilanciata non compare perché coincide con l'accuratezza.** Le pieghe della validazione B sono tutte esattamente 60 linfomi e 60 reattivi, e a classi bilanciate le due grandezze sono la stessa. Resta calcolata negli artefatti: se le pieghe si sbilanciassero, la divergenza va vista.

**L'AUC resta la metrica principale, e il motivo va detto.** La soglia decisionale è ferma a $0{,}5$ e non è mai stata tarata (§7, limitazione 4). L'AUC non dipende dalla soglia; accuratezza, precisione e F1 sì. Metterle in prima fila avrebbe penalizzato il modello per una soglia che si è deliberatamente scelto di non ottimizzare.

**Solo due voci si trasferiscono ad altre popolazioni.** Accuratezza, precisione e F1 dipendono dalla prevalenza, che qui è del $50\%$ per costruzione. Sensibilità e specificità non ne dipendono. Sono quelle due, con l'AUC, le grandezze su cui un confronto con la letteratura regge senza correzioni.

#### Le matrici di confusione

Il pannello si ricava per intero da quattro conteggi. Riportarli significa che ogni metrica è ricalcolabile da chi legge, e che non restano grandezze nascoste. Le predizioni sono fuori-piega, quindi ogni patch compare una volta sola e la matrice complessiva è la somma di quelle per piega.

**XGBoost, validazione a blocchi** (600 patch, soglia $0{,}5$):

| | Predetto FL | Predetto reattivo |
|---|:---:|:---:|
| **FL reale** | $247$ | $\mathbf{53}$ |
| **Reattivo reale** | $30$ | $270$ |

Tutti e tre i modelli, entrambe le validazioni:

| Validazione | Modello | TP | FN | FP | TN | Accuratezza |
|---|---|:---:|:---:|:---:|:---:|:---:|
| A casuale | Logistica | $253$ | $47$ | $44$ | $256$ | $0.8483$ |
| A casuale | Random Forest | $257$ | $43$ | $34$ | $266$ | $0.8717$ |
| A casuale | XGBoost | $265$ | $35$ | $30$ | $270$ | $0.8917$ |
| **B blocchi** | Logistica | $241$ | $59$ | $45$ | $255$ | $0.8267$ |
| **B blocchi** | Random Forest | $245$ | $55$ | $39$ | $261$ | $0.8433$ |
| **B blocchi** | **XGBoost** | $\mathbf{247}$ | $\mathbf{53}$ | $\mathbf{30}$ | $\mathbf{270}$ | $\mathbf{0.8617}$ |

*Nota sull'aggregazione.* Le metriche derivate da queste matrici coincidono con le medie per piega della tabella precedente, scarto nullo al quarto decimale su tutti e sei i casi, perché le pieghe hanno tutte la stessa numerosità ($120$ patch). Con pieghe di dimensione diversa le due aggregazioni divergerebbero e andrebbe dichiarato quale si riporta.

**La forbice, letta in pazienti invece che in AUC.** Su XGBoost i falsi negativi passano da $35$ con lo split casuale a $53$ con quello a blocchi. Sono **18 linfomi in più che sfuggono su 300**, e sono la stessa cosa che il $+0.0247$ di AUC di §3.1 descrive in modo più astratto. Artefatti: `confusion_matrices.csv`, `img/fase4/confusion_matrices.png`.

#### Il numero che conta clinicamente

**Il modello non riconosce circa il 18% dei linfomi follicolari**, con un intervallo che arriva al $29\%$. È la stessa prestazione descritta da «AUC $0{,}9401$», letta dal lato che conta in diagnostica: il falso negativo è un linfoma scambiato per tessuto reattivo, cioè una diagnosi mancata, mentre il falso positivo porta a un approfondimento.

Due osservazioni che il pannello rende visibili e l'AUC da sola nascondeva.

L'errore cade dal lato sbagliato. Specificità $0{,}9000$ contro sensibilità $0{,}8233$: alla soglia di $0{,}5$ il modello sbaglia più spesso mancando un linfoma che dando un falso allarme. È l'asimmetria clinicamente meno desiderabile, ed è una conseguenza della soglia non tarata, non una proprietà del modello. La curva ROC di §3.1 mostra a che prezzo in specificità si potrebbe spostare il punto di lavoro. Sceglierlo guardando questi risultati sarebbe però la stessa forma di ottimismo che la doppia validazione serve a evitare, quindi il punto riportato resta $0{,}5$.

L'incertezza sul richiamo è grande. La sensibilità oscilla fra $0{,}68$ e $0{,}94$ a seconda della piega, e l'intervallo del false negative rate va dal $6\%$ al $29\%$. La stima puntuale del $18\%$, da sola, darebbe un'idea di precisione che i dati non sostengono.

### 3.5 G. Dove sbaglia

XGBoost sbaglia **83 patch su 600** ($13{,}8\%$) nella validazione B. Gli errori **non sono sparsi**:

- **28 blocchi su 60 non contengono alcun errore**;
- **4 blocchi hanno più della metà delle patch sbagliate**;
- deviazione standard del tasso d'errore fra blocchi: $0{,}206$, contro lo $0{,}109$ atteso se gli errori fossero indipendenti dal blocco — quasi il doppio.

I cinque blocchi peggiori sono tutti di linfoma follicolare e stanno **agli estremi della numerazione**: `FL#0` ($80\%$ di errori), `FL#1` ($90\%$), `FL#27` ($40\%$), `FL#28` ($60\%$), `FL#29` ($80\%$).

**Lettura.** La concentrazione conferma che i blocchi catturano qualcosa di reale — verosimilmente casi o vetrini specifici — ed è coerente con l'ipotesi di partenza di D1. Il modello non sbaglia in modo uniforme: fallisce su un numero ristretto di casi difficili, e su quelli fallisce quasi sempre. Clinicamente è un'informazione utile quanto l'accuratezza media: un sistema di supporto che sbaglia il $14\%$ delle patch in modo casuale è diverso da uno che sbaglia sistematicamente su certi pazienti.

Non è possibile stabilire *perché* quei blocchi siano difficili senza le etichette di caso: l'analisi si ferma qui e va dichiarata come questione aperta.

### 3.6 Confronto con lo stato dell'arte

I lavori di riferimento non riportano la stessa metrica. Carreras et al. e De Souza et al. danno l'accuratezza, Sung et al. l'area sotto la curva. Le metriche sono quindi riportate **ciascuna col proprio nome e senza conversioni**: convertirne una nell'altra non è possibile, e affiancare numeri disomogenei come se fossero paragonabili produrrebbe un confronto apparente.

| Lavoro | Tecnica | Dati | Metrica riportata dagli autori | Valore |
|---|---|---|---|:---:|
| Carreras et al. (2025) | CNN ResNet end-to-end sui pixel | 221 casi, ~1,5 M patch, partizione a livello di paziente | Accuratezza patch-level | $99{,}80\%$ |
| De Souza et al. (2026) | Multimodale: CNN, vision transformer, GNN spaziale, XGBoost su morfometria | 108 casi (54 e 54), 10 centri, 4 continenti | Accuratezza, test interno | $95{,}7\%$ |
| De Souza et al. (2026) | *idem*, validazione esterna | 2 coorti indipendenti | Accuratezza, test esterno | $80{,}5\%$ e $69{,}0\%$ |
| Sung et al. (2024) | CNN interpretabile su whole slide image | metastasi linfonodali, carcinoma gastrico precoce | AUC | *compito diverso, vedi sotto* |
| **Questa tesi** | XGBoost su 33 biomarcatori interpretabili | 600 patch da ~221 casi, senza etichette di caso, validazione a blocchi | AUC / accuratezza | $0{,}9401$ / $86{,}17\%$ |

**Carreras et al.** è il lavoro da cui proviene il dataset, ed è il confronto più esposto al fraintendimento. Il $99{,}80\%$ non è commensurabile col nostro $86{,}17\%$ per quattro ragioni, che vanno citate insieme al numero: è ottenuto su circa 1,5 milioni di patch contro 600; con partizionamento a livello di paziente, che qui non è realizzabile; da una rete convolutiva end-to-end sui pixel, mentre qui si classificano 33 biomarcatori nominabili; e a una prevalenza di circa il $67\%$ di linfoma contro il nostro $50\%$, mentre l'accuratezza dipende dalla prevalenza. Il contributo di questa tesi non è competere su quel numero, ma ottenere una prestazione elevata da grandezze che un patologo può nominare e verificare.

**De Souza et al.** è il lavoro più vicino per impianto: stessa diagnosi differenziale, XGBoost su feature morfometriche, spiegabilità con SHAP, analisi spaziale dell'architettura follicolare. Ha inoltre la prevalenza bilanciata come la nostra, il che rende l'accuratezza confrontabile su questo asse. Il loro $95{,}7\%$ interno resta superiore al nostro $86{,}17\%$, con dati e modelli più ricchi. Gli autori riportano anche la validazione su due coorti esterne, $80{,}5\%$ e $69{,}0\%$, che questa tesi non ha potuto svolgere (§7, limitazione 7).

**Sung et al.** è citato come riferimento architetturale per la U-Net della Fase 2, non come termine di paragone sui risultati: il compito è la predizione di metastasi linfonodali nel carcinoma gastrico precoce, quindi un'altra malattia e un'altra domanda clinica. Confrontare la nostra AUC con la loro significherebbe confrontare due problemi diversi, e non viene fatto.

**Cosa regge il confronto e cosa no.** Sull'accuratezza il paragone è possibile solo con De Souza, ed è a nostro sfavore ma su dati e modelli non comparabili. Sensibilità e specificità sarebbero le grandezze giuste, perché non dipendono dalla prevalenza, ma né Carreras né De Souza le riportano in una forma direttamente accostabile alla nostra. Resta quindi vero, e va scritto, che **nessun confronto voce per voce con questa letteratura è pienamente pulito**.

---

## 4. Spiegabilità (SHAP)

Modello spiegato: XGBoost, scelto sulla validazione **conservativa** — sceglierlo su quella ottimistica avrebbe premiato chi sfrutta meglio il leakage.

### 4.1 D. La gerarchia SHAP concorda con la Fase 3?

| # | Biomarcatore | Importanza SHAP | Direzione | Rango in Fase 3 |
|---|---|:---:|---|:---:|
| 1 | `lbp_entropy` | $3.149$ | REACTIVE | **1** |
| 2 | `hchannel_mean` | $1.231$ | REACTIVE | 3 |
| 3 | `solidity_mean` | $0.656$ | **non monotona** | 39 |
| 4 | `glcm_contrast` | $0.588$ | REACTIVE | 25 |
| 5 | `knn1_dist_mean_um` | $0.569$ | FL | 12 |
| 6 | `n_nuclei` | $0.413$ | **non monotona** | 7 |
| 7 | `hchannel_std` | $0.330$ | **non monotona** | 37 |
| 8 | `glcm_homogeneity` | $0.285$ | FL | 9 |

**In cima l'accordo è pieno**: `lbp_entropy` è primo per entrambe le analisi, con un margine enorme ($3.15$ contro $1.23$ del secondo), e `hchannel_mean` è secondo contro terzo. La complessità della micro-tessitura cromatinica resta il discriminante principale, esattamente come concludeva la Fase 3.

Anche in fondo c'è accordo: gli ultimi cinque per SHAP (`perimeter_um_skew`, `solidity_skew`, `major_axis_um_std`, `perimeter_um_std`, `minor_axis_um_std`) sono momenti di ordine superiore, tutti oltre il 29° posto in Fase 3.

### 4.2 Due divergenze indagate

La spec (D6) impone di trattare una forte incoerenza come **sospetto di errore prima che come scoperta**. Due casi lo meritavano.

#### `solidity_mean`: terza per SHAP, trentanovesima in Fase 3 — ed è una scoperta

La Fase 3 aveva dichiarato l'intera famiglia della solidità non discriminante. Il motivo è ora chiaro:

| | FL | REACTIVE | Test |
|---|:---:|:---:|---|
| media | $0.8634$ | $0.8705$ | Mann-Whitney $p = 0{,}106$ |
| **deviazione standard** | $\mathbf{0.0303}$ | $0.0204$ | **Levene $p = 3{,}0 \times 10^{-6}$** |
| scarto medio dalla mediana | $0.0228$ | $0.0160$ | $p = 2{,}8 \times 10^{-4}$ |

Le medie sono quasi identiche — ed è tutto ciò che un test di Mann-Whitney confronta. Ma le **dispersioni** differiscono nettamente: nel linfoma follicolare la solidità nucleare è molto più **eterogenea**, con nuclei alcuni molto regolari e altri molto irregolari, mentre nel tessuto reattivo è uniforme.

È esattamente ciò che mostra il profilo SHAP per quintili ($+0.52$, $-0.64$, $-0.83$, $-0.15$, $+0.56$): una **forma a U**, con valori estremi in *entrambe* le direzioni che spingono verso il linfoma. Un effetto di questo tipo è invisibile a un confronto fra tendenze centrali.

**La forma a U è leggibile anche nella figura riassuntiva** (`img/fase4/shap_summary.png`): nella riga di `solidity_mean` i punti ad alto valore compaiono su *entrambi* i lati dello zero, mentre in `lbp_entropy` e `hchannel_mean` valori alti e valori bassi si separano nettamente ai due lati. È un indizio visivo e non una misura — l'evidenza quantitativa resta il profilo per quintili — ma rende quella figura utilizzabile in tesi per **mostrare** l'effetto di dispersione, e non soltanto per ordinare le importanze.

Non è sovradattamento: l'importanza permutazionale **fuori-piega** la colloca al 4° posto ($+0.0137$ di AUC) e la magnitudine SHAP fuori-piega ($0.632$) coincide con quella in-sample ($0.656$).

**Interpretazione clinica:** il pleomorfismo nucleare — l'eterogeneità di forma — è un tratto classico del linfoma. Il modello multivariato ha rilevato un effetto di *dispersione* che l'analisi univariata della Fase 3, costruita sul confronto fra medie, non poteva vedere. È un argomento a favore dell'approccio, non un'incoerenza.

#### `n_nuclei`: direzione non dichiarabile, e da non usare come affermazione biologica

Marginalmente il tessuto reattivo ha più nuclei ($164{,}3$ contro $149{,}2$, $p = 6{,}6 \times 10^{-14}$). Nel modello, però, il profilo per quintili è $-0.50$, $-0.43$, $+0.28$, $+0.49$, $+0.31$: cresce e poi ripiega, **non monotono**.

Verifiche svolte:

- **non è un artefatto in-sample**: il SHAP calcolato fuori-piega dà lo stesso andamento;
- **non è spiegabile per soppressione**: controllando singolarmente per ciascuna delle altre 32 variabili in un modello logistico, il coefficiente di `n_nuclei` resta negativo (da $-0.347$ a $-0.512$), mai invertito;
- **l'effetto è minuscolo**: importanza permutazionale fuori-piega $+0{,}0027$ di AUC, decimo posto.

La direzione nasce quindi da interazioni di ordine superiore dentro l'albero, non da una relazione parziale stabile. **Raccomandazione: non trarre alcuna conclusione clinica dalla direzione di `n_nuclei`.** La relazione valida da citare in tesi resta quella marginale della Fase 3 — meno nuclei nel linfoma.

### 4.3 E. Direzioni, e quando non si possono dichiarare

Fra i primi 15 biomarcatori, **quattro hanno effetto non monotono** (`solidity_mean`, `n_nuclei`, `hchannel_std`, `nuclear_area_fraction`). Per questi una direzione non è dichiarabile, e il modulo lo scrive esplicitamente invece di riportare una correlazione lineare che sarebbe falsa.

Per quelli monotoni il quadro è coerente con la Fase 3:

- `lbp_entropy` alta → **tessuto reattivo**: la cromatina del linfoma è più uniforme;
- `hchannel_mean` alto → **tessuto reattivo**;
- `glcm_contrast` alto → **tessuto reattivo**;
- `knn1_dist_mean_um` alta → **linfoma**: packing internucleare meno fitto;
- `glcm_homogeneity` alta → **linfoma**.

Tutte e cinque confermano il quadro clinico della Fase 3 su tre fronti indipendenti: nel linfoma follicolare i nuclei sono meno impaccati e la cromatina è più uniforme.

---

### 4.4 Contributo per famiglia: quale gruppo di biomarcatori regge il risultato

La classifica SHAP mette in testa `lbp_entropy` e `hchannel_mean`: **tessitura e
intensità**, non morfometria. Ma un'importanza alta non dice quanto il risultato
*dipenda* da quelle variabili. Per saperlo si rimuove una famiglia alla volta, si
riaddestra e si rimisura. Ogni sottoinsieme è valutato con lo splitter
conservativo, perché confrontare famiglie sulla stima ottimistica premierebbe
quella che sfrutta meglio il leakage.

| Sottoinsieme | n | Regr. logistica | Random Forest | XGBoost |
|---|:---:|:---:|:---:|:---:|
| Tutte | 33 | $0.899 \pm 0.073$ | $0.936 \pm 0.044$ | $0.940 \pm 0.035$ |
| Senza intensità (`hchannel_*`) | 31 | $0.901 \pm 0.070$ | $0.925 \pm 0.048$ | $0.923 \pm 0.049$ |
| Senza tessitura (GLCM, LBP) | 30 | $0.785 \pm 0.075$ | $0.876 \pm 0.026$ | $0.869 \pm 0.027$ |
| **Solo morfometria e spaziale** | 28 | $0.766 \pm 0.083$ | $0.860 \pm 0.028$ | $0.857 \pm 0.035$ |
| **Solo tessitura e intensità** | 5 | $0.891 \pm 0.087$ | $0.945 \pm 0.039$ | $0.944 \pm 0.027$ |

*(AUC-ROC, media ± deviazione standard sulle 5 pieghe della validazione a blocchi.)*

⚠️ **Come vanno letti questi numeri.** Le pieghe sono cinque, e sono le stesse
per tutti i sottoinsiemi, quindi i confronti sono appaiati. Un test appaiato non
è però in grado di dichiarare significativa alcuna differenza: con cinque coppie
le combinazioni di segno sono $2^5 = 32$, e il valore p minimo ottenibile da un
Wilcoxon a due code è $0.0625$. La scorciatoia parametrica non è praticabile
perché le pieghe condividono gran parte dell'insieme di addestramento, e per la
convalida incrociata a $k$ pieghe non esiste uno stimatore non distorto della
varianza (*Bengio & Grandvalet, 2004*), mentre i test costruiti sulle pieghe
hanno un errore di primo tipo gonfiato (*Dietterich, 1998*). Quanto segue va
quindi letto come **indicazione sostenuta dalla coerenza fra modelli**, non come
confronto dimostrato.

Il confronto appaiato è comunque informativo se si guarda **su quante pieghe** un
sottoinsieme vince, invece del solo valore p. Le vittorie sono su 5 pieghe, e
$p = 0.0625$ va letto come "tutte e cinque le pieghe concordano", cioè il massimo
di evidenza che questo disegno può produrre.

| Confronto | Logistica | Random Forest | XGBoost |
|---|:---:|:---:|:---:|
| Tutte contro senza tessitura | $+0.114$, **5/5**, $p=0.063$ | $+0.061$, **5/5**, $p=0.063$ | $+0.071$, 4/5, $p=0.125$ |
| Solo tessitura contro solo morfometria | $+0.126$, 4/5, $p=0.125$ | $+0.085$, 4/5, $p=0.125$ | $+0.087$, 4/5, $p=0.125$ |
| Tutte contro senza intensità | $-0.002$, 2/5, $p=1.000$ | $+0.011$, **5/5**, $p=0.063$ | $+0.017$, **5/5**, $p=0.063$ |
| Solo tessitura contro tutte | $-0.008$, 1/5, $p=0.438$ | $+0.009$, 4/5, $p=0.438$ | $+0.004$, 3/5, $p=0.813$ |

**Il risultato più solido è la prima riga.** Togliere la tessitura peggiora il
modello su cinque pieghe su cinque per due modelli su tre, e su quattro per il
terzo. È il massimo di concordanza ottenibile qui.

**Il divario fra le due famiglie è grande e coerente in verso.** Cinque
biomarcatori di tessitura e intensità arrivano a $0.944$ su XGBoost, i ventotto
morfometrici e spaziali si fermano a $0.857$. Sono quasi nove punti di AUC, e la
differenza va nello stesso verso in tutti e tre i modelli. Una piega su cinque va
però in direzione opposta per ciascuno di essi, il che è coerente con l'ampiezza
delle deviazioni standard e va detto.

**Quello che invece non si può affermare** è che i cinque eguaglino o superino
tutti e trentatré. Non è solo questione di ampiezza dell'effetto: **il verso
stesso non è stabile**. La logistica dice che i cinque fanno peggio (1 vittoria
su 5), Random Forest che fanno meglio (4 su 5), XGBoost è in mezzo (3 su 5). Tre
modelli che si contraddicono sulla direzione descrivono rumore, non un effetto.
La lettura corretta è che i ventotto morfometrici e spaziali **non aggiungono un
contributo misurabile** sopra i cinque, il che è diverso dal dire che valgano
meno.

**Ma non è l'intensità: è la tessitura.** Togliere `hchannel_mean` e
`hchannel_std` costa appena $0.017$ ($0.940 \rightarrow 0.923$); togliere GLCM e
LBP costa quattro volte tanto ($0.940 \rightarrow 0.869$). Il confronto appaiato
aggiunge una sfumatura: sui due modelli ad albero la perdita dovuta
all'intensità è piccola ma sistematica, presente su tutte e cinque le pieghe,
mentre sulla logistica non c'è affatto. È un effetto reale e minuscolo, non un
effetto assente. La grandezza che
porta il segnale è il **pattern della cromatina**, non quanto è scura la
colorazione — e questo è già di per sé un argomento contro l'ipotesi
dell'artefatto tecnico, perché l'intensità media è il confondente da lotto di
colorazione per eccellenza, ed è proprio quello che contribuisce meno.

#### Cosa comporta per la narrazione della tesi

Il progetto è presentato come *«paradigma white-box guidato da biomarcatori
fisici e spaziali»*. I dati dicono che a decidere è soprattutto la **tessitura
cromatinica**. Non è una smentita — il pattern della cromatina è un criterio
diagnostico che i patologi usano davvero, ed è white-box quanto l'area nucleare:
si misura, si nomina, si verifica. Ma va **nominato fra i protagonisti**, non
lasciato in coda all'elenco delle famiglie.

La morfometria non diventa inutile: da sola raggiunge $0.857$, che è un risultato
rispettabile, e resta la parte più direttamente leggibile da un clinico. La
formulazione corretta è che *le due famiglie raggiungono prestazioni diverse, e
la tessitura è quella che porta il grosso del segnale*.

Questo risultato apre però la domanda che la sezione successiva affronta: la
tessitura del canale ematossilina è anche la grandezza più esposta alla
variabilità tecnica.

**Portata di questa analisi.** La misura è interna: dice quale famiglia porta il
segnale *in questo dataset*, non che lo porterebbe altrove. Genera un'ipotesi, non
la conferma. La conferma richiederebbe una coorte indipendente, che qui non esiste
(§7, limitazione 7). Il test di §5 ne aggredisce per via sperimentale l'asse
principale, la variabilità di colorazione, ma non la sostituisce.

*Artefatti: `contribution_by_family.csv` (AUC per piega, da cui l'appaiamento è ricalcolabile), `contribution_by_family_summary.csv` (le medie della tabella sopra), `contribution_paired_tests.csv` (i confronti appaiati), tutti in `data/fase4_classification/`.*

---

## 5. Robustezza alla Variabilità di Colorazione

### 5.1 Perché questo test era necessario

L'analisi di contributo della §4.4 lascia una domanda aperta e scomoda: i biomarcatori che
portano il risultato sono quelli di **tessitura e intensità del canale
ematossilina**, che sono anche i più esposti alla variabilità *tecnica* —
lotto di colorazione, spessore della sezione, resa dello scanner. Se il modello
stesse leggendo la firma del vetrino invece della cromatina, il risultato
principale della tesi poggerebbe su un artefatto, e la validazione a blocchi
(§3.1) non basterebbe a escluderlo: i blocchi approssimano i casi, e i casi
differiscono anche per come sono stati colorati.

Il sospetto è concreto: nella decomposizione della varianza, le feature di
tessitura sono più legate al blocco di appartenenza (0.65 di varianza spiegata
entro classe) di quelle morfometriche (0.41). Ma quel dato non discrimina, perché
il blocco cattura insieme la biologia del paziente e la tecnica del vetrino.

### 5.2 Il metodo

Si perturba artificialmente la colorazione delle immagini **grezze** e si rifà
girare l'**intera** pipeline, normalizzazione di Macenko compresa. La
perturbazione è quella di Tellez et al. (2019): nello spazio delle concentrazioni
di ematossilina ed eosina, ogni canale viene alterato in modo moltiplicativo e
additivo,

$$c' = \alpha \cdot c + \beta, \qquad \alpha \sim U(1-\sigma,\ 1+\sigma), \qquad \beta \sim U(-\sigma,\ +\sigma)\cdot \overline{|c|}$$

La componente **additiva** è essenziale. La normalizzazione di Macenko riscala le
concentrazioni al 99° percentile della reference: una perturbazione solo
moltiplicativa verrebbe riassorbita quasi per intero e il test misurerebbe la
propria stessa inefficacia. La parte additiva sposta la forma della distribuzione
e sopravvive alla normalizzazione.

**La geometria resta intatta.** È la premessa che rende il test valido: se la
perturbazione danneggiasse anche la forma dei nuclei, misureremmo la robustezza a
una degradazione dell'immagine, non alla colorazione. Verificato da un test
automatico: segmentando l'immagine perturbata a $\sigma = 0.2$ si ritrovano gli
stessi nuclei con IoU $> 0.75$.

Campione: 50 patch per classe, $\sigma \in \{0,\ 0.1,\ 0.2,\ 0.3\}$.

### 5.3 Quanto si spostano i biomarcatori

Spostamento mediano, in unità di IQR del dataset — così grandezze con scale
diverse sono confrontabili. Uno spostamento di $1.0$ significherebbe che la
perturbazione ha mosso il biomarcatore quanto l'intero scarto interquartile.

| $\sigma$ | Morfometria / spaziale | Tessitura / intensità |
|:---:|:---:|:---:|
| 0.10 | 0.079 | 0.066 |
| 0.20 | 0.101 | **0.120** |
| 0.30 | 0.139 | **0.151** |

La tessitura si sposta **leggermente più** della morfometria alle perturbazioni
forti, come atteso. Ma entrambe restano molto sotto la soglia di rilevanza: anche
a $\sigma = 0.3$, che è una variazione di colorazione marcata, lo spostamento non
arriva a un sesto dell'IQR.

### 5.4 Tiene la capacità discriminante

| $\sigma$ | AUC-ROC | $\Delta p$ mediano | $\Delta p$ 90° perc. | Patch che cambiano classe |
|:---:|:---:|:---:|:---:|:---:|
| 0.00 | 1.0000 | — | — | — |
| 0.10 | 1.0000 | 0.002 | 0.074 | 1 / 100 |
| 0.20 | 0.9948 | 0.002 | 0.120 | 2 / 100 |
| 0.30 | 0.9836 | 0.004 | 0.249 | 8 / 100 |

> **Come leggere l'AUC di questa tabella.** Il valore a $\sigma = 0$ è $1.0000$
> perché il modello finale è addestrato su tutte le 600 patch, incluse queste
> 100: **il livello assoluto è privo di significato**. Ciò che il test misura è
> la *degradazione relativa*, per la quale l'ottimismo si cancella essendo lo
> stesso a numeratore e denominatore.

### 5.5 Risposta

**La tessitura sta leggendo la cromatina, non il vetrino.**

A $\sigma = 0.2$ — una variazione di colorazione che a occhio è evidente — il
modello perde mezzo punto di AUC e cambia idea su 2 patch su 100. Anche a
$\sigma = 0.3$ la perdita è di 1,6 punti e il 92% delle patch mantiene la propria
classe. Se quelle feature codificassero l'identità del vetrino, una perturbazione
di questa entità le avrebbe destabilizzate molto di più.

Il merito è in buona parte della **normalizzazione di Macenko** della Fase 1, che
qui mostra di fare esattamente il lavoro per cui è stata inserita. È un risultato
che vale anche all'indietro: giustifica a posteriori una scelta metodologica che
fino a qui era motivata solo dalla letteratura.

**Due cautele oneste.** Il 90° percentile di $\Delta p$ arriva a $0.25$ a
$\sigma = 0.3$: esiste una coda di patch — quelle già incerte, vicine alla soglia
— che si muove parecchio. E la perturbazione è *sintetica*: riproduce la
variabilità di colorazione in modo controllato, non la sostituisce. La prova
definitiva resterebbe un test su vetrini di un secondo laboratorio, che questo
dataset non permette.

*Artefatti: `data/fase4_classification/stain_robustness_*.csv`,
`img/fase4/stain_robustness.png`. Riproducibile con `python src/stain_robustness.py`.*

---

## 6. Conclusioni per la Tesi

1. **Un modello white-box su 33 biomarcatori interpretabili raggiunge AUC-ROC $0.9401$ $[0.9057,\ 0.9744]$** nella stima conservativa, con accuratezza bilanciata $0.8617$. Il risultato è ottenuto senza mai guardare i pixel: solo grandezze misurate sui nuclei segmentati.

2. **Il leakage da patch dello stesso caso valeva circa due punti di AUC**, misurati e non stimati a occhio. È una quantificazione che il dataset non permetteva di ottenere — le etichette di caso non esistono — e che la validazione a blocchi ha reso possibile.

3. **La complessità del modello conta poco, l'interpretabilità molto.** Random Forest e XGBoost sono indistinguibili fra loro; una regressione logistica arriva a $0.899$. Il valore aggiunto della fase non è nel punteggio ma nel poter dire *quali* biomarcatori decidono.

4. **`lbp_entropy` domina**, confermando in sede multivariata ciò che la Fase 3 aveva trovato con test univariati: la micro-tessitura cromatinica separa le due classi meglio di qualunque descrittore di forma o dimensione.

5. **Il pleomorfismo emerge solo in sede multivariata.** La solidità nucleare non differisce in media fra le classi, ma differisce in *dispersione*: il linfoma ha nuclei più eterogenei per regolarità di forma. È il risultato che meglio giustifica l'aggiunta di questa fase all'analisi univariata della Fase 3.

6. **A portare il segnale è la tessitura cromatinica, non la morfometria.** Cinque biomarcatori di tessitura e intensità eguagliano da soli tutti e trentatré ($0.944$ contro $0.940$); i ventotto morfometrici e spaziali si fermano a $0.857$. La formulazione «biomarcatori fisici e spaziali» va quindi corretta: la tessitura è un protagonista, non un complemento. Resta white-box a pieno titolo — il pattern della cromatina è un criterio che i patologi usano e sanno nominare — ma va presentata come tale.

7. **E quella tessitura legge la cromatina, non il vetrino.** Perturbando artificialmente la colorazione delle immagini grezze e rifacendo girare l'intera pipeline, a $\sigma = 0.2$ il modello perde mezzo punto di AUC e cambia idea su 2 patch su 100. Il merito è in buona parte della normalizzazione di Macenko della Fase 1, che qui riceve una giustificazione sperimentale a posteriori. Senza questo test, l'obiezione «state misurando il lotto di colorazione» sarebbe rimasta senza risposta — ed è l'obiezione più naturale da muovere a un risultato guidato dalla tessitura.

---

## 7. Limitazioni da Dichiarare

1. **Assenza di etichette di caso.** Il partizionamento per paziente, che Carreras et al. eseguono esplicitamente (*«hybrid partitioning… using a patient-level independent validation set»*), qui **non è possibile**: il dataset pubblicato su Zenodo è piatto e non contiene identificativi. La validazione a blocchi è un'approssimazione fondata sull'ordine di numerazione — sostenuta dall'evidenza (patch adiacenti distano $0{,}62\times$ rispetto a coppie qualsiasi; il degrado monotono di §3.2; la concentrazione degli errori di §3.5) ma **non una garanzia**.

2. **Il confronto con Carreras et al. non è diretto.** Gli autori riportano **$99{,}8\%$** di accuratezza patch-level. Il numero **non è confrontabile** con l'$0{,}9401$ di AUC di questa tesi, per tre ragioni che vanno citate insieme al confronto:
   - è ottenuto su **~1,5 milioni di patch** contro le 600 qui disponibili;
   - con **partizionamento a livello di paziente**, che qui non è realizzabile;
   - da una **CNN end-to-end sui pixel**, mentre qui si classificano 33 biomarcatori interpretabili.

   Citarlo senza queste tre precisazioni darebbe l'impressione di un divario di prestazioni, quando è in larga parte un divario di dati e di compito. Il contributo di questo lavoro non è competere su quell'accuratezza, ma ottenere una prestazione elevata da grandezze che un patologo può nominare e verificare.

3. **Numerosità.** 600 patch, e nella validazione B le unità effettivamente indipendenti sono dell'ordine delle decine (60 blocchi). Gli intervalli di confidenza sono ampi e i confronti fra modelli non raggiungono la significatività: le conclusioni di §3.3 vanno formulate come indicazioni.

4. **Soglia non ottimizzata.** La soglia decisionale resta a $0{,}5$ e non è stata scelta sui dati: ottimizzarla sul test sarebbe un'altra forma dello stesso errore che la fase intende misurare. Sensibilità e specificità a soglie diverse si leggono sulla curva ROC.

5. **Eredità dalla Fase 2: nuclei fusi, non nuclei mancanti.** I biomarcatori sono calcolati su una popolazione nucleare imperfetta, e la verifica umana di Fase 2, §3.3 ne ha chiarito la natura. Il Watershed copre l'$87{,}9\%$ dei nuclei che un lettore umano riconosce, ma il problema principale non è la rilevazione: è la **fusione di nuclei addossati**, 83 casi su 10 patch, che assorbe il $10{,}3\%$ dei nuclei marcati.

   La conseguenza sui biomarcatori è diversa da quella di una semplice perdita, e va dichiarata. Due nuclei fusi in uno producono un oggetto con area circa doppia, contorno irregolare e quindi solidità e circolarità alterate. L'effetto si propaga alle feature morfometriche (`area_um2_mean`, `solidity_*`, `circularity_*`) e a quelle di densità (`n_nuclei`), non solo ai conteggi. **L'entità di questo effetto non è stata quantificata**, e resta la principale questione aperta lasciata da questo lavoro.

6. **Blocchi difficili non spiegati.** I cinque blocchi con più errori stanno agli estremi della numerazione FL. Senza etichette di caso non è possibile stabilire se corrispondano a casi clinici particolari, a una diversa preparazione dei vetrini o ad altro.

7. **Assenza di validazione esterna.** Tutte le stime di questo lavoro, comprese quelle di §4.4, sono **interne**: provengono da una sola coorte, un solo centro, un solo scanner (Hamamatsu NanoZoomer S360) e un solo protocollo di colorazione. Prelevare altre immagini dallo stesso record Zenodo non costituirebbe una validazione esterna, perché sarebbero gli stessi casi acquisiti nelle stesse condizioni. Ne segue che i risultati **generano ipotesi e non le confermano**: la conferma richiede una coorte indipendente per centro, scanner e protocollo, che non è disponibile.

   Due elementi limitano il rischio, senza annullarlo. Il primo è il test di §5: la variabilità di colorazione e di resa dello scanner è l'asse principale su cui due coorti differiscono, ed è documentato che la firma del centro sopravvive alla normalizzazione cromatica (*Howard et al., 2021*); perturbando artificialmente quell'asse il modello perde mezzo punto di AUC a $\sigma = 0{,}2$. Il secondo è l'esito di §4.4: il segnale sta nella tessitura, mentre l'intensità media, che è il confondente da lotto di colorazione per eccellenza, è la famiglia che contribuisce meno. Sono argomenti a favore della trasferibilità, non una sua dimostrazione.

---

## 8. Riproducibilità

```bash
python src/04_classification.py     # classificazione, SHAP, contributo per famiglia
python src/stain_robustness.py      # test di robustezza alla colorazione
python src/block_structure.py       # verifica della premessa della validazione a blocchi
```

Seed $42$ ovunque; versioni fissate in `requirements.txt`. Artefatti prodotti:

| File | Contenuto |
|---|---|
| `metrics_by_model.csv` | metriche per modello × validazione × piega |
| `block_size_sensitivity.csv` | metriche al variare della dimensione del blocco |
| `block_structure_evidence.csv` | struttura conservata dall'ordine di numerazione, con test di permutazione |
| `feature_reduction.csv` | le 47 feature con gruppo, esito e rappresentante |
| `shap_importance.csv` | importanza, direzione e profilo per quintili |
| `out_of_fold_predictions.csv` | probabilità fuori-piega, per l'analisi degli errori |
| `confusion_matrices.csv` | TP, FN, FP, TN per modello e validazione, da cui il pannello è ricalcolabile |
| `best_model.joblib` | XGBoost riaddestrato su tutti i dati, con l'elenco delle feature |
| `contribution_by_family.csv` | AUC **per piega** per sottoinsieme di biomarcatori e per modello |
| `contribution_by_family_summary.csv` | media e deviazione standard delle stesse, per la tabella di §4.4 |
| `contribution_paired_tests.csv` | confronti appaiati fra sottoinsiemi, con vittorie sulle pieghe e $p$ minimo ottenibile |
| `stain_robustness_feature_shift.csv` | spostamento dei biomarcatori sotto perturbazione |
| `stain_robustness_stability.csv` | AUC e stabilità della predizione per livello di perturbazione |
| `stain_robustness_raw.csv` | biomarcatori ricalcolati, patch per patch e per ogni sigma |
| `classification_metadata.json` | seed, versioni, griglie, parametri scelti |
| `img/fase4/` | curve ROC, forbice, riepilogo SHAP, SHAP contro univariata |

Le proprietà che renderebbero falsi questi numeri sono presidiate da `tests/test_classification.py` e `tests/test_stain_robustness.py` — quest'ultimo verifica che la perturbazione cambi solo la colorazione e non la geometria dei nuclei, senza la quale il test di §5 misurerebbe un'altra cosa: in particolare l'assenza di sovrapposizione fra addestramento e test nella validazione a blocchi, il rifiuto di pieghe monoclasse (che produrrebbero un'AUC indefinita mediata in silenzio) e la dichiarazione di una direzione solo per gli effetti monotoni.

---

## 9. Bibliografia

1. **Carreras J, Ikoma H, Kikuti YY, et al.** (2025). *Histological Image Classification Between Follicular Lymphoma and Reactive Lymphoid Tissue Using Deep Learning and Explainable Artificial Intelligence (XAI)*. **Cancers**, 17(15), 2428. DOI: 10.3390/cancers17152428.
2. **Lundberg SM, Lee SI.** (2017). *A Unified Approach to Interpreting Model Predictions*. **NIPS 2017**, pp. 4765-4774. arXiv: 1705.07874. — *Fondamento teorico dei valori SHAP e della proprietà di additività verificata in §4.*
3. **Lundberg SM, Erion G, Chen H, et al.** (2020). *From local explanations to global understanding with explainable AI for trees*. **Nature Machine Intelligence**, 2, 56-67. DOI: 10.1038/s42256-019-0138-9. — *`TreeExplainer`, esatto per i modelli ad albero: è il motivo per cui le spiegazioni di §4 non sono approssimate.*
4. **Chen T, Guestrin C.** (2016). *XGBoost: A Scalable Tree Boosting System*. **KDD 2016**, pp. 785-794. DOI: 10.1145/2939672.2939785. — *Il modello selezionato dalla validazione conservativa.*
5. **Iwamoto R, Nishikawa T, Musangile FY, et al.** (2024). *Small sized centroblasts as poor prognostic factor in follicular lymphoma*. **Computers in Biology and Medicine**, 178, 108774. — *Origine dei biomarcatori `area_top10_*` sui centroblasti.*
6. **Macenko M, Niethammer M, Marron JS, et al.** (2009). *A method for normalizing histology slides for quantitative analysis*. **IEEE ISBI**, pp. 1107-1110. DOI: 10.1109/ISBI.2009.5193250. — *La normalizzazione della Fase 1, di cui §5 fornisce una giustificazione sperimentale a posteriori.*
7. **Tellez D, Litjens G, Bándi P, et al.** (2019). *Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology*. **Medical Image Analysis**, 58, 101544. DOI: 10.1016/j.media.2019.101544. — *Schema di perturbazione della colorazione ($c' = \alpha c + \beta$) adottato in §5.2.*
8. **Arlot S, Celisse A.** (2010). *A survey of cross-validation procedures for model selection*. **Statistics Surveys**, 4, 40-79. DOI: 10.1214/09-SS054. — *La rassegna di riferimento sulla convalida incrociata: fonda l'enunciato di §3.0 secondo cui la validità della procedura poggia sull'indipendenza fra addestramento e validazione.*
9. **Bussola N, Marcolini A, Maggio V, Jurman G, Furlanello C.** (2021). *AI Slipping on Tiles: Data Leakage in Digital Pathology*. In: **Pattern Recognition. ICPR International Workshops and Challenges**, LNCS 12661, pp. 167-182. Springer. DOI: 10.1007/978-3-030-68763-2_13. — *Il precedente empirico diretto: quantifica in patologia digitale il leakage da tile dello stesso soggetto, fino al $41\%$ di gonfiaggio. Termine di paragone per la forbice di $+0.0247$ misurata in §3.1.*
10. **Howard FM, Dolezal J, Kochanny S, et al.** (2021). *The impact of site-specific digital histology signatures on deep learning model accuracy and bias*. **Nature Communications**, 12(1), 4423. DOI: 10.1038/s41467-021-24698-1. — *Sostiene l'affermazione di §3.0 sulla firma del vetrino: dimostra che è apprendibile dalle immagini e che sopravvive alla normalizzazione cromatica. Motiva anche il test di robustezza di §5.*
11. **Burman P, Chow E, Nolan D.** (1994). *A cross-validatory method for dependent data*. **Biometrika**, 81(2), 351-358. DOI: 10.1093/biomet/81.2.351. — *Introduce la h-block cross-validation, di cui la validazione B di questa fase è un adattamento: è il fondamento metodologico del raggruppamento per prossimità di indice.*
12. **Roberts DR, Bahn V, Ciuti S, et al.** (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. **Ecography**, 40(8), 913-929. DOI: 10.1111/ecog.02881. — *Estende il blocking oltre le serie storiche a qualunque struttura di dipendenza, e fissa il criterio di dimensionamento del blocco usato in §3.2: deve superare la portata della dipendenza.*
13. **Racine J.** (2000). *Consistent cross-validatory model-selection for dependent data: hv-block cross-validation*. **Journal of Econometrics**, 99(1), 39-61. DOI: 10.1016/S0304-4076(00)00030-0. — *Mostra che h-block non è asintoticamente ottimale e introduce il cuscinetto hv-block. Citato in §3.0 come rimedio noto al limite di bordo del nostro schema, non applicato in questo lavoro.*
14. **Bengio Y, Grandvalet Y.** (2004). *No unbiased estimator of the variance of k-fold cross-validation*. **Journal of Machine Learning Research**, 5, 1089-1105. — *Sostiene l'avvertenza di §4.4: la varianza della convalida incrociata a $k$ pieghe non ammette stimatore non distorto, quindi un test parametrico sulle pieghe non è praticabile.*
15. **Dietterich TG.** (1998). *Approximate statistical tests for comparing supervised classification learning algorithms*. **Neural Computation**, 10(7), 1895-1923. DOI: 10.1162/089976698300017197. — *Documenta l'errore di primo tipo gonfiato dei test costruiti sulle pieghe di una convalida incrociata: è il motivo per cui §4.4 riporta indicazioni e non significatività.*
16. **De Souza LL, Chen Z, De Cáceres CVBL, et al.** (2026). *A multimodal explainable AI framework to assist in the differential diagnosis of head and neck reactive follicular hyperplasia and follicular lymphoma: an international multicentre study*. **Virchows Archiv**. DOI: 10.1007/s00428-026-04527-w. — *Il lavoro più vicino a questa tesi per compito e per impianto: stessa diagnosi differenziale, XGBoost su feature morfometriche, SHAP, analisi spaziale. Fornisce il termine di confronto sull'accuratezza a prevalenza bilanciata e, soprattutto, la misura empirica del divario fra validazione interna ed esterna citata nella limitazione 7.*
