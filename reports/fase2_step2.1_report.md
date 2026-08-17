# Report Tecnico — Fase 2 (Step 2.1): Segmentazione d'Istanza dei Nuclei Cellulari

*Modulo: [`src/02_segmentation.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/src/02_segmentation.py)*  
*Generato il 17 agosto 2026*

---

## 1. Sintesi Risultati

La segmentazione d'istanza zero-shot dei nuclei cellulari sullo Step 2.1 è stata completata con successo su tutte le 600 immagini pre-processate della Fase 1 in **53.7 secondi**.

| Metrica | Risultato complessivo |
|---------|------------------------|
| **Immagini elaborate** | **600 / 600** (300 FL + 300 REACTIVE) |
| **Nuclei cellulari totali isolati** | **94.042 nuclei** |
| **Linfoma Follicolare (FL)** | **44.749 nuclei** (media: 149.2 nuclei/patch) |
| **Tessuto Reattivo (REACTIVE)** | **49.293 nuclei** (media: 164.3 nuclei/patch) |
| **Tempo di elaborazione** | **53.7 secondi** (~11.2 immagini/secondo) |
| **File CSV Centroidi prodotto** | `data/fase2_segmentation/centroids_all.csv` (94.042 righe) |

---

## 2. Metodologia Algoritmo (Marker-Controlled Watershed)

L'algoritmo opera sui canali Ematossilina (H-channel) a 8-bit prodotti nella Fase 1:

1. **Sogliatura Adattiva / Otsu:** Separa i nuclei dallo sfondo stromatico chiaro.
2. **Trasformata di Distanza Euclidea ($EDT$):** Calcola la distanza di ogni pixel interno dal bordo del nucleo più vicino. I picchi di distanza corrispondono ai centri geometrici reali dei nuclei.
3. **Peak Local Max (Estrazione Marker):** Trova i picchi locali con una distanza minima di sicurezza tra centri adiacenti ($d_{\text{min}} = 7 \text{ px} \approx 1.61 \ \mu\text{m}$).
4. **Algoritmo Watershed Guidato da Marker:** Allaga il gradiente di distanza invertito per staccare nettamente i nuclei adiacenti a contatto.
5. **Filtraggio Morfologico per Scala Fisica:**
   - Area minima nucleare: $15 \text{ px} \approx 0.79 \ \mu\text{m}^2$ (elimina rumori residui).
   - Area massima nucleare: $1500 \text{ px} \approx 79.35 \ \mu\text{m}^2$ (elimina artefatti stromatici o sovra-aggregati).

---

## 3. Struttura degli Output Salvati

```
data/fase2_segmentation/
├── follicular_lymphoma/
│   ├── masks/                   <-- 300 maschere d'istanza (16-bit PNG, ID nucleo 1..N)
│   └── overlays/                <-- 300 immagini RGB con contorni verdi e centroidi gialli
├── reactive_tissue/
│   ├── masks/                   <-- 300 maschere d'istanza (16-bit PNG)
│   └── overlays/                <-- 300 immagini RGB con contorni verdi e centroidi gialli
├── centroids_all.csv            <-- Master CSV con coordinate (x,y) px e µm per tutti i 94.042 nuclei
├── segmentation_benchmark_preview.png
└── segmentation_metadata.json   <-- Metadati completi parametri e conteggi
```

### Struttura delle Colonne del File CSV Master (`centroids_all.csv`)

| Colonna | Tipo | Descrizione |
|---------|------|-------------|
| `image_name` | String | Nome della patch (es. `FL_examples (1)`) |
| `category` | String | Classe (`Follicular Lymphoma` / `Reactive Tissue`) |
| `nucleus_id` | Int | Identificativo numerico unico del nucleo nella patch ($1 \dots N$) |
| `centroid_y_px` | Float | Coord Y centroide in pixel ($0 \dots 223$) |
| `centroid_x_px` | Float | Coord X centroide in pixel ($0 \dots 223$) |
| `centroid_y_um` | Float | Coord Y centroide in micron ($0 \dots 51.52 \ \mu\text{m}$) |
| `centroid_x_um` | Float | Coord X centroide in micron ($0 \dots 51.52 \ \mu\text{m}$) |
| `area_px` | Int | Area del nucleo in pixel |
| `area_um2` | Float | Area reale del nucleo in $\mu\text{m}^2$ ($1 \text{ px}^2 = 0.0529 \ \mu\text{m}^2$) |

---

## 4. Evidenza Visiva

Un'anteprima comparativa a 4 campioni (2 FL e 2 REACTIVE) mostra l'eccellente precisione dell'estrazione dei contorni e dei centroidi:  
🖼️ **[segmentation_benchmark_preview.png](file:///c:/Users/Master/Desktop/testNuovoTesi/data/fase2_segmentation/segmentation_benchmark_preview.png)**

---

## 5. Prossimi Passi

- **Step 2.2:** Annotazione Ground Truth locale per la validazione quantitativa delle maschere.
- **Step 2.3:** Setup e training della U-Net ResNet-34 su PyTorch.
- **Step 2.4:** Calcolo delle metriche di confronto (Dice, IoU, AJI).
