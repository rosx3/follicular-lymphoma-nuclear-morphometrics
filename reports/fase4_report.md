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
| **Ablazione per famiglia** (§4.4) | Cinque biomarcatori di tessitura e intensità eguagliano tutti e 33 ($0.944$); i 28 morfometrici e spaziali da soli si fermano a $0.857$. **La tessitura cromatinica porta il grosso del segnale.** |
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

## 3. Risultati

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

### 3.4 F. Intervalli di confidenza

Media ± $1{,}96 \times$ errore standard sulle 5 pieghe:

| Validazione | Modello | AUC-ROC | Accuratezza bilanciata |
|---|---|:---:|:---:|
| A casuale | XGBoost | $0.9648$ $[0.9534,\ 0.9762]$ | $0.8917$ $[0.8639,\ 0.9195]$ |
| **B blocchi** | **XGBoost** | $\mathbf{0.9401}$ $[0.9057,\ 0.9744]$ | $\mathbf{0.8617}$ $[0.8156,\ 0.9077]$ |
| B blocchi | Random Forest | $0.9361$ $[0.8933,\ 0.9790]$ | $0.8433$ $[0.7903,\ 0.8964]$ |
| B blocchi | Logistica | $0.8992$ $[0.8276,\ 0.9707]$ | $0.8267$ $[0.7625,\ 0.8908]$ |

Gli intervalli della validazione B sono **tre volte più ampi** di quelli della A. Non è un difetto della validazione conservativa: è l'incertezza reale, che lo split casuale nascondeva contando come indipendenti patch che non lo sono. **Ogni valore va sempre citato col suo intervallo.**

### 3.5 G. Dove sbaglia

XGBoost sbaglia **83 patch su 600** ($13{,}8\%$) nella validazione B. Gli errori **non sono sparsi**:

- **28 blocchi su 60 non contengono alcun errore**;
- **4 blocchi hanno più della metà delle patch sbagliate**;
- deviazione standard del tasso d'errore fra blocchi: $0{,}206$, contro lo $0{,}109$ atteso se gli errori fossero indipendenti dal blocco — quasi il doppio.

I cinque blocchi peggiori sono tutti di linfoma follicolare e stanno **agli estremi della numerazione**: `FL#0` ($80\%$ di errori), `FL#1` ($90\%$), `FL#27` ($40\%$), `FL#28` ($60\%$), `FL#29` ($80\%$).

**Lettura.** La concentrazione conferma che i blocchi catturano qualcosa di reale — verosimilmente casi o vetrini specifici — ed è coerente con l'ipotesi di partenza di D1. Il modello non sbaglia in modo uniforme: fallisce su un numero ristretto di casi difficili, e su quelli fallisce quasi sempre. Clinicamente è un'informazione utile quanto l'accuratezza media: un sistema di supporto che sbaglia il $14\%$ delle patch in modo casuale è diverso da uno che sbaglia sistematicamente su certi pazienti.

Non è possibile stabilire *perché* quei blocchi siano difficili senza le etichette di caso: l'analisi si ferma qui e va dichiarata come questione aperta.

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

### 4.4 Ablazione: quale famiglia di biomarcatori regge davvero il risultato

La classifica SHAP mette in testa `lbp_entropy` e `hchannel_mean`: **tessitura e
intensità**, non morfometria. Ma un'importanza alta non dice quanto il risultato
*dipenda* da quelle variabili — solo un'ablazione lo dice. Ogni sottoinsieme è
valutato con lo splitter conservativo, perché confrontare famiglie sulla stima
ottimistica premierebbe quella che sfrutta meglio il leakage.

| Sottoinsieme | n | Regr. logistica | Random Forest | XGBoost |
|---|:---:|:---:|:---:|:---:|
| Tutte | 33 | 0.899 | 0.936 | **0.940** |
| Senza intensità (`hchannel_*`) | 31 | 0.901 | 0.925 | 0.923 |
| Senza tessitura (GLCM, LBP) | 30 | 0.785 | 0.876 | 0.869 |
| **Solo morfometria e spaziale** | 28 | 0.766 | 0.860 | **0.857** |
| **Solo tessitura e intensità** | 5 | 0.891 | 0.945 | **0.944** |

Il risultato è netto e **coerente su tutti e tre i modelli**:

**Cinque biomarcatori di tessitura e intensità eguagliano — anzi superano di un
soffio — tutti e trentatré.** I ventotto morfometrici e spaziali non aggiungono
nulla sopra di essi, e da soli si fermano a $0.857$ (XGBoost) o $0.766$
(logistica). La differenza fra $0.944$ e $0.857$ è di quasi nove punti di AUC:
non è marginale.

**Ma non è l'intensità: è la tessitura.** Togliere `hchannel_mean` e
`hchannel_std` costa appena $0.017$ ($0.940 \rightarrow 0.923$); togliere GLCM e
LBP costa quattro volte tanto ($0.940 \rightarrow 0.869$). La grandezza che
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

*Artefatto: `data/fase4_classification/ablation_by_family.csv`.*

---

## 5. Robustezza alla Variabilità di Colorazione

### 5.1 Perché questo test era necessario

L'ablazione della §4.4 lascia una domanda aperta e scomoda: i biomarcatori che
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

5. **Eredità dalla Fase 2.** I biomarcatori sono calcolati su una popolazione nucleare incompleta: la segmentazione recupera l'$85\%$ dei nuclei rilevati da Cellpose (Fase 2, §3.2). L'incompletezza è sistematica, non casuale, e potrebbe interagire con le feature di densità e distanza.

6. **Blocchi difficili non spiegati.** I cinque blocchi con più errori stanno agli estremi della numerazione FL. Senza etichette di caso non è possibile stabilire se corrispondano a casi clinici particolari, a una diversa preparazione dei vetrini o ad altro.

---

## 8. Riproducibilità

```bash
python src/04_classification.py     # classificazione, SHAP, ablazione
python src/stain_robustness.py      # test di robustezza alla colorazione
```

Seed $42$ ovunque; versioni fissate in `requirements.txt`. Artefatti prodotti:

| File | Contenuto |
|---|---|
| `metrics_by_model.csv` | metriche per modello × validazione × piega |
| `block_size_sensitivity.csv` | metriche al variare della dimensione del blocco |
| `feature_reduction.csv` | le 47 feature con gruppo, esito e rappresentante |
| `shap_importance.csv` | importanza, direzione e profilo per quintili |
| `out_of_fold_predictions.csv` | probabilità fuori-piega, per l'analisi degli errori |
| `best_model.joblib` | XGBoost riaddestrato su tutti i dati, con l'elenco delle feature |
| `ablation_by_family.csv` | AUC per sottoinsieme di biomarcatori e per modello |
| `stain_robustness_feature_shift.csv` | spostamento dei biomarcatori sotto perturbazione |
| `stain_robustness_stability.csv` | AUC e stabilità della predizione per livello di perturbazione |
| `stain_robustness_raw.csv` | biomarcatori ricalcolati, patch per patch e per ogni sigma |
| `classification_metadata.json` | seed, versioni, griglie, parametri scelti |
| `img/fase4/` | curve ROC, forbice, riepilogo SHAP, SHAP contro univariata |

Le proprietà che renderebbero falsi questi numeri sono presidiate da `tests/test_classification.py` e `tests/test_stain_robustness.py` — quest'ultimo verifica che la perturbazione cambi solo la colorazione e non la geometria dei nuclei, senza la quale il test di §5 misurerebbe un'altra cosa: in particolare l'assenza di sovrapposizione fra addestramento e test nella validazione a blocchi, il rifiuto di pieghe monoclasse (che produrrebbero un'AUC indefinita mediata in silenzio) e la dichiarazione di una direzione solo per gli effetti monotoni.

---

## 9. Bibliografia

1. **Carreras J, Ikoma H, Kikuti YY, et al.** (2025). *Histological Image Classification Between Follicular Lymphoma and Reactive Lymphoid Tissue Using Deep Learning and Explainable Artificial Intelligence (XAI)*. **Cancers**, 17(15), 2428.
2. **Lundberg SM, Lee SI.** (2017). *A Unified Approach to Interpreting Model Predictions*. **NIPS 2017**, pp. 4765-4774.
3. **Lundberg SM, Erion G, Chen H, et al.** (2020). *From local explanations to global understanding with explainable AI for trees*. **Nature Machine Intelligence**, 2, 56-67.
4. **Chen T, Guestrin C.** (2016). *XGBoost: A Scalable Tree Boosting System*. **KDD 2016**, pp. 785-794.
5. **Iwamoto R, Nishikawa T, Musangile FY, et al.** (2024). *Small sized centroblasts as poor prognostic factor in follicular lymphoma*. **Computers in Biology and Medicine**, 178, 108774.
