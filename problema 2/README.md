# Problema 2 — Estimación de Edad con CNNs

## Descripción

Red neuronal convolucional (CNN) que predice la edad de una persona a partir de una fotografía de su rostro. Es un problema de **regresión**: la salida es un número continuo (edad estimada).

---

## Estructura del proyecto

```
problema 2/
├── UTKFace/                       ← imágenes crudas (debes colocarlas aquí)
├── get_data.ipynb                 ← opcional: descarga el dataset de Kaggle
├── P2_01_dataloaders.ipynb        ← Notebook 1: pipeline de datos
├── P2_02_entrenamiento.ipynb      ← Notebook 2: arquitectura y entrenamiento
├── dataset/                       ← generado por el Notebook 1
│   ├── train/  (~16,595 imgs)
│   ├── val/    (~3,556 imgs)
│   └── test/   (~3,557 imgs)
├── dataloaders_utk.pth            ← generado por el Notebook 1
├── mejor_simple.pth               ← generado por el Notebook 2 (baseline)
└── mejor_mejor.pth                ← generado por el Notebook 2 (tu CNN)
```

> ⚠️ **Ejecuta los notebooks en orden.** El Notebook 2 carga `dataloaders_utk.pth`, generado por el Notebook 1.

---

## Dataset — UTKFace

| Atributo | Valor |
|----------|-------|
| Fuente | [Kaggle — UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) |
| Imágenes | ~23,708 fotografías de rostros |
| Resolución de trabajo | 64×64 px, 3 canales RGB |
| Etiqueta | Edad en años (entero, rango 0–116) |
| Formato del nombre | `[age]_[gender]_[race]_[datetime].jpg` |

**Ejemplo:** `25_0_2_20170116174525125.jpg` → edad = 25

### Cómo obtener el dataset

**Opción A — Manual (recomendada si ya tienes las imágenes):**

Coloca todas las imágenes directamente en la carpeta `UTKFace/`:

```
UTKFace/
  1_0_0_20161219203650533.jpg
  1_0_0_20161219222832135.jpg
  ...
```

**Opción B — Automática con `get_data.ipynb`:**

1. Crea una cuenta en [Kaggle](https://www.kaggle.com)
2. Ve a Settings → API → **Create New Token** (descarga `kaggle.json`)
3. Ejecuta `get_data.ipynb` — descarga y copia las imágenes a `UTKFace/` automáticamente

---

## Orden de ejecución

```
1. get_data.ipynb              ← solo si NO tienes UTKFace/ ya
        ↓
2. P2_01_dataloaders.ipynb     ← genera dataset/ y dataloaders_utk.pth
        ↓
3. P2_02_entrenamiento.ipynb   ← entrena SimpleCNN y MejorCNN
```

---

## Notebook 1 — DataLoaders (`P2_01_dataloaders.ipynb`)

Construye el pipeline completo de datos en PyTorch:

```
UTKFace/  →  División 70/15/15  →  AgeDataset  →  Transforms  →  DataLoader
```

### Qué hace cada parte

**`AgeDataset`**
- `__init__`: recorre la carpeta y guarda solo rutas + edades en memoria. No carga imágenes.
- `__len__`: retorna el número de ejemplos.
- `__getitem__`: carga UNA imagen del disco cuando PyTorch la pide, aplica transforms y retorna `(tensor, edad_float32)`.

**Transformaciones**
- `train_transform`: `Resize → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize`
- `val_transform`: `Resize → ToTensor → Normalize` (sin augmentation)

**DataLoaders**

| Parámetro | Train | Val / Test |
|-----------|-------|------------|
| `shuffle` | `True` | `False` |
| `drop_last` | `True` | `False` |
| `num_workers` | `0` (Windows) | `0` (Windows) |
| `pin_memory` | `True` si CUDA | `True` si CUDA |

### Archivos generados
- `dataset/train/`, `dataset/val/`, `dataset/test/`
- `dataloaders_utk.pth`

---

## Notebook 2 — Entrenamiento (`P2_02_entrenamiento.ipynb`)

### Modelo baseline — `SimpleCNN`

3 bloques convolucionales:

```
[B, 3, 64, 64]
  → Conv(3→16)  + ReLU + MaxPool  → [B, 16, 32, 32]
  → Conv(16→32) + ReLU + MaxPool  → [B, 32, 16, 16]
  → Conv(32→64) + ReLU + MaxPool  → [B, 64,  8,  8]
  → Flatten → Linear(4096→128) → ReLU → Linear(128→1)
```

### Modelo mejorado — `MejorCNN`

5 bloques convolucionales con BatchNorm y Dropout:

```
[B,   3, 64, 64]
  → Conv(3→32)   + BN + ReLU + MaxPool  → [B,  32, 32, 32]
  → Conv(32→64)  + BN + ReLU + MaxPool  → [B,  64, 16, 16]
  → Conv(64→128) + BN + ReLU + MaxPool  → [B, 128,  8,  8]
  → Conv(128→256)+ BN + ReLU + MaxPool  → [B, 256,  4,  4]
  → Conv(256→256)+ BN + ReLU + MaxPool  → [B, 256,  2,  2]
  → Flatten → Linear(1024→256) → ReLU → Dropout(0.4) → Linear(256→1)
```

### Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Épocas | 15 |
| Learning rate | 1e-3 |
| Optimizador | Adam |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Loss (backprop) | MSELoss |
| Métrica (reporte) | MAE en años |

### Archivos generados
- `mejor_simple.pth` — mejor checkpoint de SimpleCNN
- `mejor_mejor.pth` — mejor checkpoint de MejorCNN

### Tabla de experimentos

| Cambio realizado | Val MAE | Test MAE |
|------------------|---------|----------|
| SimpleCNN baseline (3 bloques) | | |
| + 2 bloques convolucionales | | |
| + BatchNorm en cada bloque | | |
| + Dropout(0.4) en el regresor | | |
| + más augmentations en train | | |

---

## Métricas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| MAE | `mean(\|ŷ − y\|)` | Error medio en años — directamente interpretable |
| RMSE | `sqrt(mean((ŷ − y)²))` | Penaliza errores grandes más que el MAE |

---

## Requisitos

```
torch >= 1.13
torchvision
Pillow
numpy
matplotlib
scikit-learn
```

Instalar con:

```bash
pip install torch torchvision Pillow numpy matplotlib scikit-learn
```

---

## Nota para Windows (VS Code)

El parámetro `num_workers` debe ser `0` en Windows para evitar errores de multiprocessing en los DataLoaders. Esto ya está configurado en los notebooks.
