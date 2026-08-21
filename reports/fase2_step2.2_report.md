# Report Tecnico — Fase 2 (Step 2.2): Dataset Ground Truth Reference

*Modulo: [`src/02_segmentation.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/src/02_segmentation.py)*  
*Generato il 17 agosto 2026*

---

> ⚠️ **Che cosa sono davvero queste maschere.** Non sono annotazioni manuali di un
> patologo: sono una **pseudo-Ground Truth** generata algoritmicamente dallo
> stesso Marker-Controlled Watershed con parametri leggermente diversi. Le
> metriche calcolate contro di esse sono quindi **inflazionate per costruzione**
> — si confronta un algoritmo con una variante di sé stesso — e non vanno citate
> come validazione indipendente. La validazione indipendente è quella contro la
> Ground Truth Cellpose, in [`fase2_report.md`](file:///c:/Users/Master/Desktop/testNuovoTesi/reports/fase2_report.md) §3, con maschere
> archiviate in `data/ground_truth/cellpose_v4/`. Il limite è discusso per esteso
> nella nota metodologica in testa a [`src/02_segmentation.py`](file:///c:/Users/Master/Desktop/testNuovoTesi/src/02_segmentation.py).

## 1. Sintesi Risultati

Lo **Step 2.2 (Creazione del Dataset Ground Truth Reference)** è stato completato con successo. È stato estratto e validato un subset benchmark di **30 patch rappresentative** (15 Linfoma Follicolare + 15 Tessuto Reattivo) con campionamento stratificato basato sulla densità cellulare.

| Parametro | Valore |
|-----------|--------|
| **Patch Ground Truth totali** | **30 patch** |
| **Patch Linfoma Follicolare (FL)** | 15 patch (range densità: **66 – 206 nuclei/patch**) |
| **Patch Tessuto Reattivo (REACTIVE)** | 15 patch (range densità: **75 – 227 nuclei/patch**) |
| **Campionamento** | Stratificato per densità (5 a bassa densità, 5 a media, 5 ad alta) |
| **Directory Maschere GT** | `data/ground_truth/` |
| **Directory Preview Visive** | `img/fase2/gt_benchmark_preview.png` |

---

## 2. Anteprima Visiva a 6 Livelli di Densità

L'anteprima grafica comparativa mostra i 3 terzili di densità (bassa, media, alta densità) per entrambe le classi:  
🖼️ **[`img/fase2/gt_benchmark_preview.png`](file:///c:/Users/Master/Desktop/testNuovoTesi/img/fase2/gt_benchmark_preview.png)**
