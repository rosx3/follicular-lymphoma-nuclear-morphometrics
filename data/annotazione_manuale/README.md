# Annotazione manuale di riferimento — istruzioni

> ⚠️ **Prima di cominciare, non aprire:** `immagini_metadata.json`, la cartella
> `data/ground_truth/cellpose_v4/`, e qualunque figura con i contorni
> sovrapposti. Contengono le risposte degli algoritmi. Vederle prima di annotare
> ti orienterebbe verso di esse senza che te ne accorga, e renderebbe inutile
> tutto il lavoro.

## Cosa è questa annotazione, e cosa non è

Va detto subito, perché determina come il risultato potrà essere presentato.

**Nessuna delle persone coinvolte in questo lavoro è un patologo.** L'annotazione
che segue è quindi un **riferimento umano non esperto**, e come tale va sempre
etichettata: non è una ground truth clinica, e non può esserlo.

Questo non la rende inutile, perché il compito richiesto non è diagnostico. Non
si chiede di dire se un tessuto è linfoma o iperplasia reattiva, si chiede di
individuare corpi cromatinici distinti. È un compito percettivo, con regole
scritte, molto meno dipendente dalla formazione specialistica.

Quello che l'annotazione può stabilire:
- se Cellpose stia frammentando un nucleo in più oggetti
- se stia contando come nuclei cose che nuclei non sono
- quale fra Cellpose e Watershed sia più vicino a un conteggio umano

Quello che **non** può stabilire: quale sia il conteggio nucleare corretto in
senso clinico. Per quello servirebbe un patologo, e non è disponibile.

## Cosa devi fare

Su ciascuna delle 10 immagini, **un clic al centro di ogni nucleo**.
Nient'altro. Non servono contorni.

Le immagini sono le patch intere, ingrandite 4 volte: 896×896 px sullo schermo
per 224×224 px reali. Sono tutte e sole le patch da cui provengono i numeri
pubblicati del benchmark, quindi annotandole tutte il riferimento copre l'intero
insieme di valutazione.

Conta circa 15 minuti a immagine, quindi due ore e mezza abbondanti in totale.
Non farle tutte di fila: l'attenzione cala e il conteggio ne risente.

## Le regole, da fissare adesso e non cambiare

Queste decisioni vanno prese **prima** del primo clic. Deciderle mentre annoti
rende il riferimento non riproducibile, nemmeno da te a distanza di tempo.

**Cosa conta come nucleo.** Un nucleo cellulare, riconoscibile come corpo
cromatinico distinto.

**Cosa non conta, ed è da escludere sempre:**
- eritrociti, che non hanno nucleo
- detriti e frammenti senza struttura cromatinica riconoscibile
- artefatti di taglio e di colorazione

**Nuclei sul bordo dell'immagine.** Si contano se il **centro** cade dentro
l'immagine. Se il centro è fuori e vedi solo un lembo, non si conta. È la stessa
convenzione che applicano gli algoritmi con cui verrai confrontata.

**Nuclei sovrapposti o addossati.** **Questo è il punto su cui i due algoritmi
divergono di più**, quindi è quello su cui il giudizio umano conta davvero.

> **Regola effettivamente applicata** (fissata sulla prima immagine, il 2
> settembre 2026, e da mantenere su tutte le altre): *due clic quando il contorno
> dei due nuclei è distinguibile, uno solo quando si vede una massa unica senza
> contorno riconoscibile.*
>
> Va riportata insieme ai risultati. Una regola più permissiva darebbe conteggi
> più alti e una più restrittiva più bassi: senza dichiararla, il numero di
> nuclei umani non è interpretabile.

**Nuclei pallidi o sfocati.**

> **Regola effettivamente applicata** (accertata sulle prime quattro immagini, il
> 2 settembre 2026): *i nuclei pallidi sono stati esclusi, e non marcati come
> dubbi.* La categoria "dubbio" è stata usata per altri casi, non per la
> scarsa visibilità.
>
> Ne segue che **il conteggio umano è un limite inferiore**: conta i nuclei ben
> visibili, non tutti i nuclei presenti. Va dichiarato ogni volta che il numero
> viene riportato.
>
> Conseguenza sulle metriche, da tenere presente leggendo i risultati:
> - **recall e fusioni restano validi**, perché si calcolano solo sui punti
>   marcati e non dipendono dalla soglia di visibilità;
> - il conteggio delle **istanze senza punto umano non è interpretabile**, perché
>   mescola nuclei pallidi esclusi di proposito con eventuali falsi positivi
>   dell'algoritmo, e non li distingue.
>
> Mantenere questa stessa soglia su tutte e dieci le immagini. Cambiarla a metà
> renderebbe i conteggi non confrontabili fra loro.

## I casi dubbi

Non forzarti a decidere. Tienili in un file separato.

La percentuale di dubbi **è un risultato**, non un fallimento: dice quanto il
compito sia intrinsecamente ambiguo, e quindi quanto sia ragionevole aspettarsi
che due algoritmi divergano fra loro.

## Come farlo in Fiji

1. `File > Open` sull'immagine.
2. `Image > Properties`: assicurati che **Unit of length** sia `pixel` e che
   Pixel width e Pixel height siano `1`. Con una calibrazione attiva le
   coordinate uscirebbero in un'altra unità e non tornerebbero.
3. `Analyze > Set Measurements`: spunta **Centroid**.
4. Seleziona lo strumento **Multi-point** (clic destro prolungato sullo
   strumento punto).
5. Clicca al centro di ogni nucleo. Un clic sbagliato si toglie con `Alt` più
   clic sul punto.
6. `Analyze > Measure` (`Ctrl+M`): compare la tabella con le colonne `X` e `Y`.
7. `File > Save As > Results...` e salva come **`<nome_immagine>_punti.csv`**
   nella stessa cartella.
8. Poi aggiungi i casi dubbi **senza deselezionare i punti già messi**, rifai
   `Ctrl+M` e salva come **`<nome_immagine>_dubbi.csv`**. Il secondo file sarà
   quindi un sovrainsieme del primo: contiene i certi più i dubbi. Va benissimo,
   lo script ricava i dubbi per differenza.
   **Salva sempre prima i certi e poi i dubbi, mai al contrario:** invertendo
   l'ordine i due file si scambierebbero di significato senza che nulla lo
   segnali. Se non ci sono dubbi, il secondo file non serve.

Esempio di nomi attesi:

```
FL_examples_20_punti.csv
FL_examples_20_dubbi.csv
```

Suggerimento pratico: per non perdere il segno su immagini con quasi duecento
nuclei, procedi a bande orizzontali dall'alto verso il basso, invece che a caso.

## Il secondo lettore

Serve a misurare quanto le regole scritte sopra siano applicabili in modo
consistente. Senza questo dato, il riferimento umano appare come una verità
assoluta, e non lo è.

Non essendo disponibile un patologo, le opzioni praticabili sono due, e vanno
dichiarate per quello che sono.

**Ripetizione a distanza, da parte tua.** Rifai due o tre immagini dopo qualche
settimana, senza riguardare i CSV precedenti. Misura la variabilità
intra-osservatore, è una misura standard, non richiede nessun'altra persona e non
ha bisogno di essere presentata come qualcosa che non è.

**Secondo lettore indipendente, non esperto.** Una seconda persona che annota le
stesse due o tre immagini seguendo questo stesso documento. Misura la
riproducibilità del protocollo, cioè se le regole bastino a sé stesse. Va
riportata con questo nome, mai come *variabilità inter-osservatore* in senso
clinico.

Le due si combinano bene. Salva i file con suffissi distinti, per esempio
`_punti_A.csv`, `_punti_B.csv`, `_punti_A2.csv` per la tua ripetizione.

## Cosa succede dopo

Con i CSV si calcola quanti dei tuoi nuclei ciascun algoritmo trova, quanti ne
inventa, e quindi se sia Cellpose a sovra-segmentare o il Watershed a
sotto-segmentare. Oggi è una domanda senza risposta.

## Nota tecnica sulle coordinate

Le immagini sono le patch originali 224×224 ingrandite 4 volte. La conversione è

```
x_patch = x_click / 4
y_patch = y_click / 4
```

Non devi farla tu: la applica lo script di confronto.
