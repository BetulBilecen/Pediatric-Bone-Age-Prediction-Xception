# 🦴 Pediatric Bone Age Estimation from Hand X-Rays
### Multi-Input Regression with Xception Transfer Learning | TensorFlow & Keras

---

## 📌 Overview

This project builds a deep learning **regression model** that estimates pediatric bone age (in months) from hand X-ray images. It uses a **Multi-Input Architecture** that combines:

- **Image branch:** Xception CNN (pretrained on ImageNet, fully fine-tuned) extracts visual features from the X-ray
- **Tabular branch:** A small Dense network encodes patient gender (0/1)
- **Fusion head:** Both branches are concatenated and passed through fully connected layers to predict bone age

The model also includes **Grad-CAM visualization** (`explainability.py`) to highlight which bone regions the model focuses on when making predictions.

---

## 📦 Dataset

**Source:** [RSNA Pediatric Bone Age Challenge — Kaggle](https://www.kaggle.com/datasets/kmader/rsna-bone-age)  
**Task:** Continuous regression — predict bone age in **months**  
**Format:** Grayscale PNG images, resized to 128×128 px (RGB for Xception compatibility)

| Split      | Size   |
|------------|--------|
| Train      | ~10,088|
| Validation | ~1,261 |
| Test       | ~1,261 |

> **Note:** The original Kaggle test set has no labels. The test split used here is carved from the training CSV via an 80/10/10 split using `train_test_split`.

---

## 🏗️ Model Architecture

A **Functional API multi-input model** combining image and gender data:

```
Image Input (128, 128, 3)          Gender Input (1,)
        │                                  │
  Xception Base                      Dense(16, relu)
  (ImageNet weights,                       │
   fully fine-tuned)                       │
        │                                  │
GlobalMaxPooling2D (2048,)                 │
        │                                  │
        └──────── Concatenate (2064,) ─────┘
                        │
                  Dense(32, relu)
                        │
               Dense(1, linear)  →  Predicted Bone Age (months)
```

| Detail               | Value                                   |
|----------------------|-----------------------------------------|
| Base Model           | Xception (ImageNet weights)             |
| Fine-Tuning          | ✅ Full (`base_model.trainable = True`) |
| Pooling              | GlobalMaxPooling2D                      |
| Gender Encoding      | Dense(16, relu)                         |
| Fusion               | Concatenate → Dense(32) → Dense(1)      |
| Output Activation    | Linear (regression)                     |
| Loss Function        | MSE                                     |
| Metrics              | MAE                                     |
| Total Parameters     | ~20.9M (79.83 MB)                       |

### Why Multi-Input?
Gender is a clinically significant factor in bone development — bone maturation rates differ between males and females. Adding gender as a dedicated input branch (rather than ignoring it) gives the model direct access to this signal without forcing it to infer it from pixel data.

---

## ⚙️ Training Configuration

| Hyperparameter     | Value                         |
|--------------------|-------------------------------|
| Optimizer          | Adam (lr = 0.0001)            |
| Loss Function      | MSE                           |
| Max Epochs         | 15                            |
| **Actual Epochs**  | **14** (EarlyStopping triggered) |
| Batch Size         | 32                            |
| Image Size         | 128×128                       |
| EarlyStopping      | patience=5, monitors val_loss, restores best weights |
| ModelCheckpoint    | saves best val_loss only      |

---

## 🔄 Data Augmentation

Applied **only to training data** via `ImageDataGenerator`. Validation uses `preprocess_input` only (no augmentation).

| Technique         | Value         |
|-------------------|---------------|
| Rotation          | ±20°          |
| Zoom              | 15%           |
| Horizontal Flip   | ✅ Enabled    |
| Preprocessing     | `preprocess_input` (normalizes to [−1, 1]) |

> ⚠️ **No `rescale=1/255`** — Xception's `preprocess_input` already handles pixel normalization. Applying both would corrupt the input distribution.

---

## 📊 Results

### Training History

| Epoch | Train Loss (MSE) | Train MAE | Val Loss (MSE) | Val MAE  |
|-------|-----------------|-----------|----------------|----------|
| 1     | 1867.68         | 28.74     | 414.84         | 16.25    |
| 2     | 388.63          | 15.54     | 357.53         | 14.67    |
| 3     | 333.64          | 14.36     | 311.16         | 13.59    |
| 4     | 306.19          | 13.82     | 269.53         | 12.97    |
| 5     | 281.44          | 13.25     | 293.53         | 13.38    |
| 6     | 267.94          | 12.86     | 326.95         | 14.43    |
| 7     | 248.47          | 12.45     | 322.12         | 14.56    |
| 8     | 232.39          | 12.01     | 237.26         | 11.99    |
| 9     | 220.20          | 11.71     | 236.14         | **12.09** |
| 10    | 209.22          | 11.39     | 268.90         | 12.96    |
| 11    | 203.45          | 11.24     | 236.69         | 12.17    |
| 12    | 187.49          | 10.78     | 300.56         | 13.95    |
| 13    | 182.10          | 10.65     | 272.55         | 13.18    |
| 14    | 171.89          | 10.31     | 329.51         | 14.47    |

> Best checkpoint: **Epoch 9** → Val MSE: **236.14**, Val MAE: **~12.09 months**

### Final Performance

| Metric         | Value                    |
|----------------|--------------------------|
| Best Val MSE   | **236.14**               |
| Best Val MAE   | **~12.09 months**        |
| Baseline MAE   | 13.0 months (single-input Xception) |
| Improvement    | **~0.91 months** over baseline ✅ |

Adding gender as a second input improved MAE by approximately **0.91 months** over the single-input baseline, confirming that multi-input fusion is beneficial.

---

## 🔥 Grad-CAM Visualization

The project includes a Grad-CAM module (`explainability.py`) that overlays attention heatmaps on X-ray images to show which regions the model focuses on for each prediction.

### Figure — Grad-CAM Attention Map
> *Example: Actual bone age 150 months → Predicted 144.8 months*

The model correctly focuses on the **carpal bones and metacarpal growth plates** — anatomically consistent with radiological bone age assessment. The attention map confirms the model has learned clinically meaningful features rather than image artifacts.

---

## 🗂️ Project Structure

```
📦 Pediatric-Bone-Age-Prediction-Xception/
├── src/
│   ├── main.py                    # Training pipeline entry point
│   ├── models_architecture.py     # Multi-input Xception model definition
│   ├── dataset.py                 # Data loading, preprocessing, generators
│   ├── explainability.py          # Grad-CAM heatmap visualization
│   └── models/
│       └── best_xception_multi_input.h5  # Best checkpoint (saved during training)
├── bonage_dataset/
│   ├── boneage-training-dataset/
│   ├── boneage-test-dataset/
│   ├── boneage-training-dataset.csv
│   ├── boneage-test-dataset.csv
│   └── df_test_split.csv          # Auto-generated test split after training
├── Images/
│   ├── Prediction_Results.png
│   ├── Sample_Hand_XRay.png
│   └── sample_xray.png
├── main.py                        # Root-level entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Setup & Usage

### Requirements

```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib pillow opencv-python
```

### Dataset Setup

After downloading from Kaggle, ensure PNG files sit **directly** inside their dataset folders:

```
bonage_dataset/
├── boneage-training-dataset/
│   ├── 1377.png
│   └── ...
├── boneage-training-dataset.csv
└── boneage-test-dataset.csv
```

### Train

```bash
python src/main.py
```

Training will print a **Baseline Comparison Report** at the end, comparing the multi-input model's best val MAE against the 13.0-month single-input baseline.

### Grad-CAM Inference

```bash
python src/explainability.py
```

Randomly selects a test sample from `df_test_split.csv` and displays the original X-ray alongside its Grad-CAM attention map.

---

## ⚠️ Known Limitations

**Image resolution trade-off** — 128×128 was used to reduce training time. Upscaling to 256×256 with a GPU may recover fine-grained spatial detail (subtle growth plate features) and improve accuracy.

**Minimal fusion head** — The head uses `Dense(32) → Dense(1)`. A larger head (`Dense(256) → Dense(64) → Dense(1)`) may better map the 2064-dim fused representation to continuous bone age values.

**Large epoch-1 loss spike** — Training all Xception weights from scratch causes a large initial loss (MSE ~1867). Gradual fine-tuning (freeze base → train head → unfreeze) would reduce this instability.

**MSE as loss** — MSE heavily penalizes outliers. Switching to MAE as the loss function may produce more clinically stable results, especially for extreme age values.

---

## 🔧 Suggested Next Steps

1. **Increase image resolution** to 256×256 with GPU for richer spatial features
2. **Gradual fine-tuning** — train head first with frozen base, then unfreeze Xception layer by layer
3. **Larger regression head** — `Dense(256) → Dense(64) → Dense(1)` for richer feature mapping
4. **MAE loss** — more robust to outlier age samples than MSE
5. **Test set evaluation** — use saved `df_test_split.csv` to measure final held-out performance with the best checkpoint

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Xception-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Grad--CAM-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-train__test__split-lightgrey)
![Dataset](https://img.shields.io/badge/Dataset-RSNA_Bone_Age-lightgrey)

---

> **Disclaimer:** This project is for educational purposes only and is not intended for clinical use.
