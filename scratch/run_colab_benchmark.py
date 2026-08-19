"""
===============================================================================
Script autonomo per Google Colab (GPU T4) — v3 (Diametro Calibrato 22 px)
Esegue il Benchmark Fase 2 con GT Cellpose indipendente + U-Net ResNet-34
===============================================================================
"""

import os
import cv2
import csv
import json
import time
import zipfile
import numpy as np
import importlib.util
from pathlib import Path

# 1. Estrarre il file ZIP se non ancora estratto
if os.path.exists('colab_benchmark.zip') and not os.path.exists('02_segmentation.py'):
    with zipfile.ZipFile('colab_benchmark.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    print('[OK] Archivio estratto.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from cellpose import models as cp_models

# Import dinamico di 02_segmentation.py
spec = importlib.util.spec_from_file_location('segmentation', '02_segmentation.py')
seg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seg)

# Configurazione Device (GPU T4 su Colab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device in uso: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')

BASE_DIR = Path('.')
FASE1_DIR = BASE_DIR / 'data' / 'fase1_preprocessing'

# Rilevamento dinamico delle patch effettivamente estratte da colab_benchmark.zip
fl_h_dir = FASE1_DIR / 'follicular_lymphoma' / 'h_channel'
re_h_dir = FASE1_DIR / 'reactive_tissue' / 'h_channel'

fl_files = sorted([f.replace('_hchannel.png', '') for f in os.listdir(fl_h_dir) if f.endswith('_hchannel.png')])
re_files = sorted([f.replace('_hchannel.png', '') for f in os.listdir(re_h_dir) if f.endswith('_hchannel.png')])

PATCHES = [{'name': f, 'category': 'Follicular Lymphoma'} for f in fl_files] + \
          [{'name': f, 'category': 'Reactive Tissue'} for f in re_files]

print(f'[Rilevate] {len(fl_files)} patch FL + {len(re_files)} patch RE (totale: {len(PATCHES)})')

# Split 20 train / 10 val (stratificato 5 FL + 5 RE val)
rng = np.random.default_rng(42)
fl_p = [p for p in PATCHES if p['category'] == 'Follicular Lymphoma']
re_p = [p for p in PATCHES if p['category'] == 'Reactive Tissue']

fl_idx = rng.permutation(len(fl_p))
re_idx = rng.permutation(len(re_p))

n_val_fl = min(5, len(fl_p) // 3)
n_val_re = min(5, len(re_p) // 3)

val_patches   = [fl_p[i] for i in fl_idx[:n_val_fl]] + [re_p[i] for i in re_idx[:n_val_re]]
train_patches = [fl_p[i] for i in fl_idx[n_val_fl:]] + [re_p[i] for i in re_idx[n_val_re:]]

print(f'[Split] Train: {len(train_patches)} patch ({len(train_patches)-n_val_re} FL + {n_val_re} RE)')
print(f'[Split] Val:   {len(val_patches)} patch ({n_val_fl} FL + {n_val_re} RE)')

# 1. Generazione GT Cellpose (su GPU con diametro calibrato a 22 px ≈ 5 µm per linfociti)
CALIBRATED_DIAMETER_PX = 22.0  # 22 px -> 10.1 µm con la calibrazione rivista (vedi src/calibration.py)
print(f'\n[Cellpose] Generazione GT indipendente (diametro calibrato = {CALIBRATED_DIAMETER_PX} px ≈ 5 µm)...')
cp_gpu = torch.cuda.is_available()
cp_model = cp_models.CellposeModel(gpu=cp_gpu, model_type='nuclei')

cellpose_gt = {}
t0_cp = time.time()
for p in val_patches:
    name = p['name']
    cat = 'follicular_lymphoma' if p['category'] == 'Follicular Lymphoma' else 'reactive_tissue'
    h_path = FASE1_DIR / cat / 'h_channel' / f'{name}_hchannel.png'
    h_img = cv2.imread(str(h_path), cv2.IMREAD_GRAYSCALE)
    if h_img is None:
        raise FileNotFoundError(f'Impossibile leggere: {h_path}')
    
    masks, _, _ = cp_model.eval(h_img, diameter=CALIBRATED_DIAMETER_PX, channels=[0,0], flow_threshold=0.4, cellprob_threshold=0.0)
    cellpose_gt[name] = masks.astype(np.int32)
    print(f'  [Val GT] {name[:30]:30s} -> {masks.max()} nuclei trovati da Cellpose')

print(f'[Cellpose] GT generata in {time.time()-t0_cp:.1f} secondi su {"GPU" if cp_gpu else "CPU"}.')

# Genera maschere Watershed per i train patch per addestrare U-Net
ws_train_masks = {}
for p in train_patches:
    name = p['name']
    cat = 'follicular_lymphoma' if p['category'] == 'Follicular Lymphoma' else 'reactive_tissue'
    h_path = FASE1_DIR / cat / 'h_channel' / f'{name}_hchannel.png'
    h_img = cv2.imread(str(h_path), cv2.IMREAD_GRAYSCALE)
    labels, _ = seg.segment_nuclei_watershed(h_img)
    ws_train_masks[name] = labels

# 2. Addestramento U-Net su GPU
class ColabDataset(Dataset):
    def __init__(self, patches, masks_dict):
        self.patches = patches
        self.masks_dict = masks_dict
        self.to_tensor = transforms.ToTensor()
    def __len__(self):
        return len(self.patches)
    def __getitem__(self, idx):
        p = self.patches[idx]
        name = p['name']
        cat = 'follicular_lymphoma' if p['category'] == 'Follicular Lymphoma' else 'reactive_tissue'
        rgb_path = FASE1_DIR / cat / 'rgb_normalized' / f'{name}_norm.png'
        img = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        img_t = self.to_tensor(img)
        mask = (self.masks_dict[name] > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return img_t, mask_t

train_ds = ColabDataset(train_patches, ws_train_masks)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

model = seg.UNetResNet34(num_classes=1, pretrained=True, decoder_dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
bce_loss = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    total = (probs + targets).sum(dim=(1, 2, 3))
    return 1.0 - (2.0 * inter + eps) / (total + eps)

print('\n[U-Net] Training su GPU (15 epoche)...')
t0_un = time.time()
model.train()
for epoch in range(1, 16):
    epoch_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = bce_loss(logits, masks) + dice_loss(logits, masks).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 5 == 0 or epoch == 1:
        print(f'  Epoca {epoch:02d}/15 loss={epoch_loss/len(train_loader):.4f}')

print(f'[U-Net] Addestramento completato in {time.time()-t0_un:.1f} secondi.')

# 3. Valutazione Benchmark vs GT Cellpose
results = []
model.eval()

for p in val_patches:
    name = p['name']
    cat = p['category']
    cat_dir = 'follicular_lymphoma' if cat == 'Follicular Lymphoma' else 'reactive_tissue'
    gt_mask = cellpose_gt[name]
    
    # Watershed
    h_path = FASE1_DIR / cat_dir / 'h_channel' / f'{name}_hchannel.png'
    h_img = cv2.imread(str(h_path), cv2.IMREAD_GRAYSCALE)
    ws_mask, _ = seg.segment_nuclei_watershed(h_img)
    
    # U-Net
    rgb_path = FASE1_DIR / cat_dir / 'rgb_normalized' / f'{name}_norm.png'
    rgb_img = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    img_t = transforms.ToTensor()(rgb_img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(img_t)).squeeze().cpu().numpy()
    prob_u8 = (prob * 255).astype(np.uint8)
    _, bin_un = cv2.threshold(prob_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    un_mask, _ = seg.segment_nuclei_watershed(bin_un)
    
    ws_m = seg.compute_segmentation_metrics(gt_mask, ws_mask)
    un_m = seg.compute_segmentation_metrics(gt_mask, un_mask)
    
    results.append({
        'image_name': name, 'category': cat,
        'ws_dice': ws_m['dice'], 'ws_iou': ws_m['iou'], 'ws_aji': ws_m['aji'], 'ws_f1_det': ws_m['f1_det'],
        'un_dice': un_m['dice'], 'un_iou': un_m['iou'], 'un_aji': un_m['aji'], 'un_f1_det': un_m['f1_det']
    })

# Stampa tabella finale
ws_d = np.mean([r['ws_dice'] for r in results])
ws_a = np.mean([r['ws_aji'] for r in results])
ws_f = np.mean([r['ws_f1_det'] for r in results])

un_d = np.mean([r['un_dice'] for r in results])
un_a = np.mean([r['un_aji'] for r in results])
un_f = np.mean([r['un_f1_det'] for r in results])

print('\n' + '='*55)
print(f' RISULTATI BENCHMARK COLAB (GT: Cellpose diameter={CALIBRATED_DIAMETER_PX}px, val n=10)')
print('='*55)
print(f'{"Metrica":<22} {"Watershed":>14} {"U-Net ResNet34":>14}')
print('-'*55)
print(f'{"Dice (pixel-level)":<22} {ws_d:>14.4f} {un_d:>14.4f}')
print(f'{"AJI (instance-level)":<22} {ws_a:>14.4f} {un_a:>14.4f}')
print(f'{"F1 Detection @0.5":<22} {ws_f:>14.4f} {un_f:>14.4f}')
print('='*55)

# Salva CSV
with open('colab_benchmark_results.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
print('\n[SUCCESS] File "colab_benchmark_results.csv" generato con successo!')
