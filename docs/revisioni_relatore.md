# Revisioni del relatore — domande aperte e azioni

Elenco delle domande poste dal relatore sul progetto, con la risposta verificata e
l'azione che ne consegue. Serve a separare due cose: chiarire un punto e
modificare la pipeline. Le domande si chiariscono subito, le modifiche si
raggruppano e si eseguono in blocco.

Legenda stato: `da fare` · `in corso` · `fatto` · `chiusa senza modifiche`

---

## 1. L'input della Fase 2 è l'immagine RGB o quella in bianco e nero?

**Data:** 2 settembre 2026
**Stato risposta:** chiarita
**Stato azione:** da fare

### Risposta verificata

Nessuna delle due. L'input della segmentazione è il **canale Ematossilina
(H-channel)**: un'immagine a un solo canale, ottenuta per deconvoluzione
colorimetrica dall'RGB normalizzata, non per desaturazione.

- La Fase 1 salva due prodotti: `<nome>_norm.png` (RGB, 3 canali) e
  `<nome>_hchannel.png` (canale H, 1 canale). Vedi `src/run_pipeline.py:200`.
- La Fase 2 legge solo il secondo, con `cv2.IMREAD_GRAYSCALE`, e lo passa a
  `segment_nuclei_watershed()`. Vedi `src/run_pipeline.py:272` e `:277`.
- La binarizzazione vera e propria (bianco/nero puro) avviene **dentro** la
  Fase 2, come primo passo del Watershed: soglia di Otsu in
  `src/02_segmentation.py:209`.
- L'RGB resta usata per gli overlay visivi e per la U-Net di confronto.

### Azione che ne deriva: confronto a parità di input fra Watershed e U-Net

> Nota terminologica (2 settembre 2026): la parola "ablazione" non va usata nei
> documenti destinati alla tesi. Il relatore la associa alla validazione su una
> coorte esterna, e l'equivoco è già costato una discussione. Vedi la voce 4.

**Problema.** Oggi il confronto non è a parità di input. Il Watershed lavora sul
canale H, la U-Net sull'RGB a 3 canali
(`scratch/run_colab_benchmark.py:170`). L'asimmetria favorisce la U-Net, che
riceve più informazione e in più è addestrata sul dominio. La conclusione della
tesi resta quindi conservativa. Ma un revisore non è tenuto a fidarsi
dell'argomento: serve un numero.

**Proposta.** Non sostituire il braccio RGB, aggiungerne uno. Riportare tre
bracci, tutti valutati contro la stessa GT Cellpose:

1. Watershed su canale H, zero-shot.
2. U-Net su RGB normalizzata (caso realistico migliore per una CNN).
3. U-Net su canale H replicato su 3 canali (confronto a parità di input).

La differenza fra 2 e 3 misura il contributo dell'informazione dell'eosina alla
segmentazione nucleare, che è un risultato citabile di per sé.

**Modifica tecnica.** Una riga nel `ColabDataset` di
`scratch/run_colab_benchmark.py`: leggere `*_hchannel.png` invece di
`*_norm.png` e replicarlo con `np.stack([h] * 3)`, perché l'encoder ResNet-34
pre-addestrato richiede 3 canali in ingresso. Costo di calcolo trascurabile: 20
patch, 15 epoche, batch 4.

**Vincoli da rispettare.**
- Servono almeno 3 seed per braccio. La U-Net ha varianza run-to-run già
  documentata in `reports/fase2_report.md`, sezione 7.7.5. Con un solo seed si
  rischia di leggere come effetto del canale quello che è rumore di
  inizializzazione.
- Con n = 10 patch di validazione il test non ha potenza per distinguere
  effetti piccoli. Va dichiarato prima di vedere i numeri, non dopo.
- La riesecuzione richiede GPU (Colab) e un riaddestramento completo, non un
  semplice ri-scoring. Vedi `reports/fase2_report.md`, sezione 7.8.

**Punto separato da dichiarare comunque in tesi.** Il target di addestramento
della U-Net è la maschera prodotta dal Watershed
(`scratch/run_colab_benchmark.py:155`). La rete non impara a segmentare i
nuclei, impara a imitare il Watershed. Questa asimmetria è più rilevante di
quella sul canale e va detta esplicitamente, a prescindere dall'esito
del confronto.

**Prima di implementare:** aprire un brainstorming per fissare la spec (seed,
formato della tabella a tre bracci, collocazione nel report di Fase 2).

---

## 2. In base a cosa si scelgono le immagini da mandare alla U-Net?

**Data:** 2 settembre 2026
**Stato risposta:** chiarita
**Stato azione:** da fare

### Risposta verificata

La selezione avviene su tre livelli. Solo il primo segue un criterio esplicito,
gli altri due sono un sorteggio.

**Livello 1, da 600 immagini a 30.** Campionamento stratificato per densità
cellulare: 5 patch a bassa densità, 5 a media e 5 ad alta, per ciascuna delle due
classi. Range da 66 a 227 nuclei per patch. Documentato in
`data/ground_truth/gt_metadata.json`, campo `stratification_strategy`. Il
razionale è coprire l'intervallo di difficoltà della segmentazione: le patch
dense sono quelle dove il Watershed rischia di fondere nuclei adiacenti.

**Livello 2, da 30 a 20 train + 10 val.** Permutazione casuale con seed fisso 42,
in `scratch/run_colab_benchmark.py:101`. Bilanciata per classe (5 FL + 5 RE in
validazione, sempre). La densità non entra nel criterio.

**Livello 3, cosa vede la U-Net.** Le 20 patch di training. Le altre 10 servono a
valutare entrambi i metodi. Su questo il confronto è corretto: Watershed e U-Net
sono giudicati sulle stesse 10 patch.

### Problema rilevato: la stratificazione non sopravvive allo split

Rieseguendo la permutazione con seed 42 e incrociandola con le densità di
`gt_metadata.json`:

| | Bassa densità | Media | Alta | Nuclei/patch (media) |
|---|:---:|:---:|:---:|:---:|
| Validazione (10) | 2 | 5 | 3 | 188.2 |
| Training (20) | 8 | 5 | 7 | 167.1 |

Il set di validazione è sbilanciato verso l'alta densità. Le due patch più rade in
assoluto (71 e 76 nuclei) finiscono entrambe in training. La patch meno densa
della validazione ne ha 152. Il benchmark misura quindi le prestazioni nel regime
medio-alto, e il regime rado non viene mai messo alla prova.

Non è un errore grave, ed è l'esito atteso pescando 10 elementi a caso da 30. Ma
svuota di significato lo sforzo fatto al Livello 1: si stratifica il campione per
densità e poi si rompe la stratificazione con un sorteggio.

### Azione 2A: split stratificato o multi-split

Due opzioni, in ordine di preferenza.

1. **Ripetere il benchmark su più split diversi** e riportare media e dispersione
   invece del risultato di un singolo sorteggio. A n = 10 l'esito dipende
   parecchio da quale split è uscito, e questa opzione lo rende esplicito invece
   di nasconderlo.
2. **Stratificare lo split anche per strato di densità**, oltre che per classe.
   Più semplice, ma resta un singolo split e quindi un singolo campione.

Le due si combinano bene: split stratificati per densità, ripetuti su più seed.
Da decidere insieme al confronto del punto 1, perché si riesegue lo stesso
benchmark su Colab e conviene farlo una volta sola.

### Azione 2B: unificare le due implementazioni dello split

`src/02_segmentation.py:650` contiene `split_gt_patches()`, documentata, coperta
da `tests/test_segmentation_split.py` e con l'avviso metodologico sul data
leakage. **Non è la funzione usata.** Lo script Colab reimplementa lo split
inline. Il percorso coperto dai test non è quello che ha prodotto i numeri della
tesi.

Da correggere facendo usare allo script la funzione del modulo. Attenzione: le
due implementazioni non sono equivalenti (`split_gt_patches` usa
`val_fraction`, lo script usa `min(5, len // 3)`), quindi il cambio modifica lo
split e impone di rieseguire il benchmark.

### Azione 2C: correggere la contraddizione interna al report — FATTA (2 settembre 2026)

Il problema. `reports/fase2_report.md` sezione 6.1 affermava che le metriche
della U-Net "restano inferiori a quelle del Watershed (Dice 57.4% vs 63.7%)".
Erano numeri della versione v3 del benchmark. La sezione 3.2 dello stesso report
riporta i numeri v4, dove la U-Net sta marginalmente sopra: 0.8038 contro
0.7950. Le due sezioni si contraddicevano.

La correzione. Sottosezione riscritta, ora intitolata "Perché l'Impianto
Sperimentale Penalizza la U-Net". Riporta i numeri v4, dichiara la correzione in
modo esplicito con la data, e mantiene le due ragioni strutturali per cui il
disegno sfavorisce la U-Net (target circolare, regime small-data). La
conclusione è allineata alla Sezione 4: parità a costo nullo, nessuna
superiorità rivendicata.

Verificato che nessun'altra affermazione nelle sezioni 1-6 citi i numeri v3. Le
occorrenze rimaste (sezioni 7.7 e 7.8, righe 254, 322, 351) sono storiche e
correttamente etichettate come tali.

---

## 3. Come si giustifica la suddivisione in blocchi contigui?

**Data:** 2 settembre 2026
**Stato risposta:** chiarita
**Stato azione:** 3A e 3B fatte, 3C e 3D da fare

### Risposta verificata

La giustificazione è già nella spec di Fase 4, decisione D1, e si articola in
quattro passi. Vanno presentati in quest'ordine, perché il punto non è "perché
blocchi" ma "perché non si poteva fare di meglio".

1. **Premessa.** Le 600 patch non vengono da 600 pazienti. La serie di origine
   (Carreras et al. 2025) è di 221 casi, 177 FL e 44 reattivi, da cui sono state
   estratte circa 1,5 milioni di patch. Più patch provengono quindi dallo stesso
   vetrino.
2. **Conseguenza.** Con uno split casuale il modello può imparare la firma del
   vetrino, cioè intensità della colorazione, resa dello scanner, spessore della
   sezione, invece della biologia. Le feature più esposte sono `hchannel_mean`,
   `glcm_contrast`, `glcm_homogeneity` e `lbp_entropy`.
3. **Perché non si usa lo standard.** Lo standard sarebbe `GroupKFold`
   sull'identificativo di caso. Il dataset Zenodo è piatto: due cartelle da 300
   file, nessun identificativo di caso. L'informazione non è recuperabile.
4. **Il sostituto.** L'ordine di numerazione dei file. Se l'esportazione è
   avvenuta caso per caso, patch con indici vicini tendono a venire dallo stesso
   vetrino.

### Il punto che rende difendibile tutto il resto

L'ipotesi del passo 4 non è assunta, è misurata. Rieseguita il 2 settembre 2026
e riprodotta al quarto decimale:

| Classe | Coppie adiacenti | Coppie qualsiasi | Rapporto |
|---|:---:|:---:|:---:|
| Linfoma follicolare | 5.6689 | 9.2153 | **0.6152** |
| Tessuto reattivo | 6.3894 | 9.2529 | **0.6905** |

Test di permutazione, 2000 estrazioni: p = 0.0005 in entrambe le classi, cioè il
minimo rappresentabile con quel numero di estrazioni. Nessuna permutazione
casuale dell'ordine raggiunge un rapporto basso quanto quello osservato.

### Perché la dimensione del blocco non è arbitraria

Due argomenti, ed è la domanda che segue immediatamente.

Il valore principale, 10, è stato scelto **prima** di vedere le metriche.
Deriva dall'ordine di grandezza della serie di origine: 300 patch reattive per
44 casi danno circa 7 patch per caso, e 10 aggiunge margine.

La conclusione non dipende da quel valore. L'analisi di sensibilità su blocchi
da 5, 10, 20 e 30 mostra un degrado monotono su tutti e tre i modelli. È
l'argomento decisivo: se i blocchi non catturassero struttura reale, la curva
sarebbe piatta.

### Perché l'errore, se c'è, è dalla parte giusta

Per il linfoma follicolare ci sono 177 casi su 300 patch, quindi circa 1,7 patch
per caso. Blocchi da 10 raggruppano insieme casi distinti. È un raggruppamento
eccessivo, non insufficiente, e raggruppare troppo può solo togliere segnale,
mai aggiungere leakage. La stima che ne esce è un limite inferiore.

### Come formularlo in tesi

- Non chiamare mai i blocchi "casi" o "pazienti". Sono una sonda. Formulazione
  corretta: raggruppamento per prossimità di indice, usato come approssimazione
  conservativa dell'appartenenza allo stesso caso.
- Riportare la forbice, non il numero migliore. Il leakage valeva circa due
  punti di AUC, 0.9648 contro 0.9401 su XGBoost. Che sia stato misurato invece
  che assunto è di per sé un contributo metodologico.
- Dichiarare il limite residuo: se l'ordine di esportazione non fosse
  strettamente contiguo per caso, blocchi da 10 potrebbero ancora spezzare un
  caso fra addestramento e test, e la validazione B non sarebbe pienamente
  conservativa. Non è verificabile senza le etichette.

### Azione 3A: rendere riproducibile la verifica — FATTA (2 settembre 2026)

Il problema. Il rapporto 0.615 / 0.691 è il pilastro dell'intera
giustificazione, ma nel repository esisteva solo come affermazione, nella spec e
nell'intestazione di `04_classification.py`. Nessuno script lo produceva.

La soluzione. Nuovo modulo `src/block_structure.py`, sulla falsariga di
`src/stain_robustness.py`: funzioni pure più un `main()` che scrive
`data/fase4_classification/block_structure_evidence.csv`. Si esegue con
`python src/block_structure.py`. Aggiunge al valore già citato un test di
permutazione con stimatore (1 + successi) / (1 + permutazioni), che evita di
riportare un p pari a zero (Phipson & Smyth, 2010).

Coperto da `tests/test_block_structure.py`, 8 test. I due centrali verificano
che la misura segnali la struttura quando c'è e non la segnali quando non c'è,
su matrici sintetiche. Suite completa: 213 test passati.

Aggiornati anche il puntatore nell'intestazione di `src/04_classification.py` e
la tabella degli artefatti in `reports/fase4_report.md`.

### Azione 3B: portare la giustificazione dentro il report — FATTA (2 settembre 2026)

Il problema. Il ragionamento viveva nella spec di progetto, che è un documento
interno. Il report di Fase 4 lo dava per acquisito e mostrava solo i risultati.
In più la bibliografia della fase, sette voci, non conteneva nulla sulla
convalida sotto dipendenza: l'approccio non era ancorato a nessuna letteratura.

La soluzione. Nuova sottosezione §3.0 in `reports/fase4_report.md`, "Perché due
validazioni, e perché blocchi contigui". La sezione 3 è stata rinominata
"Validazione e Risultati" per coerenza. Contiene la premessa, il rischio
documentato nel dominio, il metodo con il suo nome, la verifica empirica
dell'ipotesi e il limite del nostro adattamento.

Sei riferimenti aggiunti in bibliografia, voci 8-13, tutti con DOI verificato
su Crossref o sulla pagina dell'editore, e ciascuno con la glossa che dice quale
affermazione sostiene:

| Voce | Sostiene |
|---|---|
| Arlot & Celisse (2010) | la convalida incrociata presuppone indipendenza |
| Bussola et al. (2021) | il leakage da tile dello stesso soggetto, fino al 41% |
| Howard et al. (2021) | la firma del vetrino è apprendibile e sopravvive alla normalizzazione |
| Burman, Chow & Nolan (1994) | h-block cross-validation, il metodo di cui la validazione B è un adattamento |
| Roberts et al. (2017) | il blocking oltre le serie storiche, e il criterio di dimensionamento |
| Racine (2000) | hv-block, rimedio noto al limite di bordo |

Erano cinque nel piano iniziale. Arlot & Celisse era stata scartata come
ridondante, ma scrivendo il testo il primo enunciato della sezione è rimasto
senza fonte, e per la regola del relatore ogni affermazione ne vuole una.

**Limite nuovo, emerso scrivendo.** h-block rimuove un intorno attorno a ogni
punto di test. `GroupKFold` su blocchi interi lascia invece i vicini immediati
delle patch di bordo nell'insieme di addestramento: con blocchi da 10 riguarda 2
patch su 10. Il nostro schema è quindi leggermente meno severo di un h-block
vero. Dichiarato in §3.0, rimedio noto citato, non applicato.

### Azione 3C: stimare la portata della dipendenza

Roberts et al. (2017) stabiliscono che il blocco deve superare la portata della
dipendenza. Quella portata si può stimare dai dati: basta calcolare il rapporto
di somiglianza non solo fra patch adiacenti, ma in funzione della distanza di
indice, e vedere a quale distanza la curva risale al valore globale. È l'analogo
del variogramma in statistica spaziale, ed è un'estensione naturale di
`src/block_structure.py`.

Il guadagno è nella formulazione della tesi. Oggi si scrive "blocco da 10 scelto
a priori con un margine di prudenza". Con la stima si scriverebbe "blocco da 10
scelto a priori e successivamente verificato maggiore della portata di
dipendenza stimata". La seconda chiude la domanda sulla dimensione del blocco
invece di rispondervi con un'analisi di sensibilità.

### Azione 3D (facoltativa): cuscinetto hv-block

Applicare davvero il cuscinetto, cioè escludere dall'addestramento le patch
immediatamente adiacenti al confine del blocco di test. Chiuderebbe il limite di
bordo invece di dichiararlo. Costo: una riesecuzione della Fase 4 e un terzo
valore da riportare accanto ad A e B. Da valutare solo se 3C non basta.

---

## 4. «L'ablazione si fa con un altro dataset, e nel nostro caso non è possibile»

**Data:** 2 settembre 2026
**Stato risposta:** chiarita
**Stato azione:** fatta

### Il nodo: due parole per due cose diverse

Quello che il relatore descrive è la **validazione esterna**, cioè rimisurare su
una coorte indipendente per centro, scanner e protocollo. Risponde alla domanda
*il risultato tiene altrove?* e richiede effettivamente un secondo dataset.

L'analisi fatta in §4.4 è invece una **rimozione di componenti**: si toglie una
famiglia di biomarcatori, si riaddestra, si rimisura sugli stessi dati.
Risponde a *quale pezzo regge il risultato?* e non richiede altri dati. Usa solo
`features_patches_master.csv`, che esiste ed è completo.

Sulla fattibilità l'obiezione quindi non si applica. Sul merito, però, il
relatore ha ragione su due punti, ed è quello che conta.

### I due punti su cui ha ragione

**La validazione esterna non è possibile**, e non solo per mancanza di un file.
Le 600 patch vengono da una sola coorte, un solo centro, un solo scanner
Hamamatsu NanoZoomer S360, un solo protocollo di colorazione. Prendere altre
immagini dallo stesso record Zenodo non sarebbe validazione esterna: stessi
casi, stesse condizioni di acquisizione.

**Un'analisi interna genera ipotesi, non le conferma.** Coincide con il limite
di potenza statistica già noto: 5 pieghe, circa 60 blocchi indipendenti, e un
Wilcoxon appaiato su 5 coppie che non può scendere sotto p = 0.0625.

### Decisione: la parola non si usa più

Su richiesta, il termine "ablazione" è stato eliminato da tutti i documenti e
dal codice. L'equivoco lessicale non va lasciato in circolo, perché si
ripresenterebbe in sede di discussione. La formula adottata è **"analisi del
contributo per famiglia di biomarcatori"**.

### Cosa è stato fatto (2 settembre 2026)

**Rinomina completa.** `reports/fase4_report.md` (§1, §4.4, §4.2, §8),
`README.md` (3 punti), `src/04_classification.py`
(`feature_family_ablation` → `feature_family_contribution`, commenti, etichetta
interna, stampa), `src/stain_robustness.py`, `tests/test_classification.py` (2
test rinominati). Artefatto rinominato con `git mv`:
`ablation_by_family.csv` → `contribution_by_family.csv`. Zero occorrenze
residue del termine nel repository, esclusa questa nota. Suite: 213 test
passati.

**§4.4 riscritta.** Tolta l'affermazione che cinque biomarcatori "eguagliano,
anzi superano" tutti e trentatré: su XGBoost la differenza è 0.9439 contro
0.9401, con deviazioni standard di 0.027 e 0.035, quindi dentro il rumore. La
tabella ora riporta le deviazioni standard, che prima non comparivano. Aggiunto
un avviso su come vanno letti i numeri, con il limite del Wilcoxon a 5 coppie e
il motivo per cui la scorciatoia parametrica non è praticabile
(*Bengio & Grandvalet, 2004*; *Dietterich, 1998*, voci 14 e 15 in bibliografia,
DOI verificati). Quel che resta affermato è il divario fra le due famiglie,
0.944 contro 0.857 su XGBoost, coerente per verso su tutti e tre i modelli:
indicazione, non dimostrazione.

**Limitazione 7 aggiunta.** Assenza di validazione esterna. La sezione 7 ne
aveva sei e questa mancava del tutto, pur essendo la prima che un revisore cerca
in un lavoro clinico. Dichiara il limite e indica le due mitigazioni parziali:
il test di robustezza alla colorazione di §5, che aggredisce l'asse principale
su cui due coorti differiscono, e l'esito di §4.4, per cui il segnale sta nella
tessitura mentre l'intensità media, confondente da lotto di colorazione per
eccellenza, è la famiglia che contribuisce meno.

### Azione 4A: conservare le AUC per piega — FATTA (2 settembre 2026)

Il problema. `feature_family_contribution()` calcolava le AUC per piega e poi le
scartava, salvando solo media e deviazione standard. Il confronto appaiato non
era quindi ricalcolabile dagli artefatti, ed era il presupposto per poter
scrivere qualunque cosa sulla significatività.

La soluzione. La funzione ora restituisce la tabella per piega, allineandosi alla
convenzione già usata da `metrics_by_model.csv` e `block_size_sensitivity.csv`.
`summarise_contribution()` produce la vista aggregata con `ddof=0`, cioè
esattamente i numeri già pubblicati: rigenerati e verificati identici al quarto
decimale su tutte e 15 le righe, quindi il refactor non ha spostato nulla.

Aggiunta anche `paired_family_tests()`, che esegue i confronti appaiati sui
quattro accostamenti fissati in `FAMILY_COMPARISONS`, scelti prima di guardare i
risultati. Riporta differenza media, vittorie sulle pieghe, p di Wilcoxon e p
minimo ottenibile. Tre nuovi artefatti in `data/fase4_classification/`. Suite:
216 test passati, tre nuovi.

**Cosa ha rivelato il test appena è stato possibile eseguirlo.** L'affermazione
più solida non era quella che il report metteva in evidenza. Togliere la
tessitura peggiora il modello su 5 pieghe su 5 per logistica e Random Forest, e
su 4 su 5 per XGBoost: è la concordanza massima ottenibile con questo disegno.
Il confronto fra i cinque biomarcatori e tutti e trentatré, invece, non ha verso
stabile: 1 vittoria su 5 per la logistica, 4 su 5 per Random Forest, 3 su 5 per
XGBoost. Tre modelli che si contraddicono sulla direzione descrivono rumore. È
un argomento più forte di quello usato prima, che si limitava all'ampiezza
dell'effetto rispetto alla deviazione standard. §4.4 aggiornata di conseguenza.

### Rimane da fare, se si vuole andare oltre l'indicazione

Aumentare il numero di ricampionamenti, perché con 5 pieghe il p minimo di un
Wilcoxon appaiato è 0.0625 e nessuna differenza può risultare significativa a
0.05. Due strade: `GroupKFold` ripetuto su più partizionamenti, oppure
leave-one-block-out, che con circa 60 blocchi darebbe 60 stime invece di 5.
Con la tabella per piega ora conservata, il calcolo a valle è già pronto.

---

## 5. «Perché una sola metrica? E non è confrontabile con la letteratura»

**Data:** 2 settembre 2026
**Stato risposta:** chiarita
**Stato azione:** fatta

### Risposta verificata

Le metriche calcolate erano quattro, non una: AUC, accuratezza bilanciata,
sensibilità e specificità, per piega, in `metrics_by_model.csv`. Ma il report ne
mostrava una sola, e sensibilità e specificità non comparivano in nessuna
tabella. Sulla presentazione l'osservazione era quindi fondata.

L'AUC era in prima fila per una ragione mai scritta: la soglia decisionale è
ferma a 0,5 e deliberatamente non tarata (limitazione 4). L'AUC non dipende dalla
soglia, accuratezza, precisione e F1 sì. Metterle in testa avrebbe penalizzato il
modello per una soglia che si è scelto di non ottimizzare. Ora la ragione è nel
report.

### Cosa è stato fatto (2 settembre 2026)

**Pannello in `_fold_metrics()`.** AUC, accuratezza, accuratezza bilanciata,
precisione, richiamo, specificità, F1, false negative rate e false positive rate.
Aggiunto lì perché è il punto da cui passano tutte le tabelle prodotte da
`evaluate()`: metriche per modello, sensibilità alla dimensione del blocco e
contributo per famiglia lo ereditano senza altri interventi.

Set scelto dall'utente sulla base dei lavori citati (AUC, accuratezza,
precisione, richiamo, F1), più due aggiunte motivate:

- **specificità**, perché delle cinque solo il richiamo non dipende dalla
  prevalenza, e sensibilità con specificità è l'unica coppia che si trasferisce
  fra popolazioni con prevalenze diverse;
- **false negative rate**, chiesto dal relatore. È $1 -$ richiamo, quindi non
  aggiunge informazione: aggiunge la lettura clinica corretta.

Scartate PR-AUC (utile solo a classi sbilanciate), MCC (segue l'accuratezza a
classi bilanciate), valore predittivo negativo (raddoppia la riserva sulla
prevalenza senza aggiungere nulla) e Brier score (misura la calibrazione, non la
discriminazione: serve solo se la probabilità mostrata dalla GUI entra nella
tesi).

**Fase 4 rieseguita per intero.** Tutti i numeri già pubblicati invariati al
quarto decimale. 218 test passati, due nuovi: uno fissa le identità che devono
valere esattamente (FNR = 1 − sensibilità, FPR = 1 − specificità, F1 media
armonica), l'altro fissa l'ipotesi delle pieghe bilanciate sotto cui accuratezza
e accuratezza bilanciata coincidono.

**§3.4 riscritta** come pannello completo con intervalli di confidenza, più la
lettura clinica. **§3.6 «Confronto con lo stato dell'arte»** aggiunta, che il
relatore aveva chiesto: un lavoro per riga con tecnica, dati e metrica riportata
col proprio nome, senza conversioni. De Souza et al. (2026) aggiunto in
bibliografia come voce 16.

### Il numero che ne è uscito

Il modello non riconosce circa il **18% dei linfomi** (FNR 0,1767, intervallo da
0,063 a 0,291). È la stessa prestazione di «AUC 0,9401» letta dal lato che conta
in diagnostica. E l'errore cade dal lato sbagliato: specificità 0,90 contro
sensibilità 0,82, quindi alla soglia di 0,5 il modello manca un linfoma più
spesso di quanto dia un falso allarme.

**Questione aperta.** Spostare il punto di lavoro sulla ROC ridurrebbe il FNR a
scapito della specificità. Sceglierlo guardando questi risultati sarebbe però lo
stesso ottimismo che la doppia validazione serve a evitare. Da decidere: se
riportare un punto alternativo dichiarandolo illustrativo e non validato, oppure
lasciare solo 0,5.

### Nota di metodo emersa qui

Il relatore aveva suggerito di appoggiare la limitazione 7 al fatto che De Souza,
validando all'esterno, scende da 95,7% a 80,5% e 69,0%. **Scartato su richiesta
dell'utente:** giustificare ciò che non si è potuto fare dicendo che ad altri è
andata peggio non è un argomento. In §3.6 resta il dato fattuale, cioè che loro
la validazione esterna l'hanno fatta e questa tesi no.

---

## 6. Verifica umana del riferimento, e rimozione della U-Net

**Data:** 2 settembre 2026
**Stato:** fatta

### Da dove è nata

Rispondendo alla domanda sui blocchi contigui è emerso, verificando il codice,
che il confronto Watershed contro U-Net non opponeva due paradigmi:

- la U-Net era addestrata sulle maschere prodotte dal Watershed stesso
  (`scratch/run_colab_benchmark.py:155`), quindi ne imitava il comportamento;
- il suo output binario veniva comunque separato in istanze **dal Watershed**
  (riga 231), con gli stessi parametri.

Il Watershed girava quindi dentro entrambi i bracci, e il confronto misurava
solo due rilevatori di primo piano a valle della stessa macchina.

### Annotazione manuale (1656 nuclei)

Restava aperta una domanda più grossa: Cellpose, che fa da metro a tutta la
Fase 2, non era mai stato verificato su queste immagini.

Sono state annotate a mano, alla cieca, tutte e 10 le patch di validazione:
**1656 nuclei**, 64 dubbi (3,9%), circa 10 minuti a immagine. Protocollo in
`data/annotazione_manuale/README.md`, analisi in `src/annotation_agreement.py`.

| | Watershed | Cellpose |
|---|:---:|:---:|
| Nuclei umani coperti | 87,9% | **95,2%** |
| Fusioni di nuclei distinti | **83** | 2 |
| Immagini con almeno una fusione | **10/10** | 2/10 |
| Nuclei assorbiti in istanze fuse | **170 (10,3%)** | 4 (0,2%) |

Confronti appaiati su n = 10: Cellpose vince in 10 immagini su 10 su entrambe le
grandezze, p = 0,002, il minimo ottenibile.

**Tre conclusioni.** Il riferimento regge, quindi l'impianto di validazione è
salvo. Il limite del Watershed non è la rilevazione ma la separazione. E il
conteggio complessivo del Watershed coincide con quello umano (rapporto mediano
1,001) solo perché due errori opposti si compensano: un confronto sui soli
conteggi avrebbe concluso per un accordo perfetto.

**Limiti dichiarati.** Il lettore non è un patologo, e non ne esiste uno nel
progetto: il riferimento è non esperto. I nuclei pallidi sono stati esclusi, quindi
il conteggio umano è un limite inferiore, e per questo si riportano solo recall e
fusioni, che non dipendono da quella soglia.

### Cosa è stato modificato

**U-Net rimossa** da sintesi, tabella dei risultati, breakdown per classe e
conclusioni della Fase 2, e da tutto il README. Le sezioni 6.1, 7.7 e 7.8 la
citano ancora ma sono marcate come storiche. La classe `UNetResNet34` resta nel
codice con un'intestazione che spiega perché non è usata.

**Conclusione 1 riscritta:** non più «un algoritmo zero-shot eguaglia una rete
addestrata», ma la scelta di un metodo deterministico per vincolo (nessuna
annotazione disponibile) sostenuta da Malpica (1997) e Veta (2013), quest'ultimo
descrive uno schema quasi identico al nostro. Aggiunto Malpica in bibliografia
con DOI verificato.

**Nuova §3.3 in Fase 2** con metodo, limiti, risultati e conclusioni.
**Limitazione 5 della Fase 4 riscritta**: non più «popolazione nucleare
incompleta» ma «nuclei fusi», con la conseguenza sui biomarcatori morfometrici.

**Matrici di confusione aggiunte** (§3.4 di Fase 4), per tutti i modelli e
entrambe le validazioni, più la figura. Verificato che le metriche derivate
coincidano con le medie per piega: scarto nullo, perché le pieghe hanno tutte
120 patch.

### Conseguenze sulla lista

Le voci **1, 2A e 2B si chiudono senza lavoro**: riguardavano lo split
train/val, che esisteva solo per addestrare la U-Net. Senza rete non serve
nessuno split.

**Non resta più niente che richieda la GPU.**

### Rimane aperto

L'effetto delle fusioni sui biomarcatori non è quantificato. Il 10,3% dei nuclei
è coinvolto, e una fusione non è un nucleo perso: è un oggetto con area doppia e
forma alterata, che entra nelle feature morfometriche. Dichiarato nella
limitazione 5, non misurato.

---

## Minore, non urgente

- `scratch/run_colab_benchmark.py:115`: la riga di log stampa la composizione
  dello split in modo errato (calcola le patch FL di train come
  `len(train_patches) - n_val_re`). Non tocca i risultati, solo il messaggio a
  video, ma confonde chi rilegge il log del Colab.
