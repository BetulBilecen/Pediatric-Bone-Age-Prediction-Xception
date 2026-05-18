# 🦴 Pediatric Bone Age Estimation from Hand X-Rays
### Regression with Xception Transfer Learning | TensorFlow & Keras

---

## 📌 Overview

This project builds a deep learning **regression model** that estimates pediatric bone age (in months) from hand X-ray images. Rather than training from scratch, it leverages **Xception** — a state-of-the-art CNN architecture pretrained on ImageNet — and applies **Transfer Learning with Full Fine-Tuning** to adapt it to skeletal development patterns in radiographic images.

---

## 📦 Dataset

**Source:** [RSNA Pediatric Bone Age Challenge — Kaggle](https://www.kaggle.com/datasets/kmader/rsna-bone-age)  
**Task:** Continuous regression — predict bone age in **months**  
**Format:** Grayscale PNG images, resized to 128×128 px (RGB mode for Xception compatibility)

| Split      | Source                    | Size  |
|------------|---------------------------|-------|
| Train      | 80% of training CSV       | ~9,600|
| Validation | 10% of training CSV       | ~1,200|
| Test       | 10% of training CSV       | ~1,200|

> **Note:** The original Kaggle test CSV does not include ground-truth bone age labels, so the test set used for evaluation here is carved out from the training dataset using an 80/10/10 split via `train_test_split`.

### Figure 4 — Sample Hand X-Ray (Training Set)
![Sample Hand X-Ray](Figure_4_Sample_XRay.png)

> A representative grayscale hand radiograph from the training dataset. The model learns to estimate bone age from structural cues such as bone density, growth plate width, and carpal bone development visible in images like this.

---

## 🏗️ Model Architecture

Xception base (pretrained on ImageNet) with a custom regression head:

```
Input: (128, 128, 3)
│
├── Xception Base (Fine-Tuned, all layers trainable)
│     └── Output: (4, 4, 2048)
│
├── GlobalMaxPooling2D   →  (2048,)
├── Dense(5)
├── ReLU Activation
└── Dense(1, linear)     →  Predicted bone age in months
```

| Detail               | Value                        |
|----------------------|------------------------------|
| Base Model           | Xception (ImageNet weights)  |
| Fine-Tuning          | ✅ Full (`model.trainable = True`) |
| Pooling              | GlobalMaxPooling2D           |
| Output Activation    | Linear (regression)          |
| Loss Function        | Mean Squared Error (MSE)     |
| Metrics              | MAE, MSE                     |

### Why Xception?
- **Depthwise Separable Convolutions** give strong feature extraction with fewer parameters than VGG or ResNet
- Proven performance on medical imaging tasks
- Effective at capturing fine bone structure and growth plate details in X-rays
- Excellent generalization when fine-tuned on domain-specific data

---

## ⚙️ Training Configuration

| Hyperparameter      | Value                  | Note                                      |
|---------------------|------------------------|-------------------------------------------|
| Optimizer           | Adam                   |                                           |
| Learning Rate       | 0.0001                 | Lowered from 0.001 for stable fine-tuning |
| Loss Function       | MSE                    |                                           |
| Max Epochs          | 25                     |                                           |
| **Actual Epochs**   | **16**                 | EarlyStopping triggered                   |
| Batch Size          | 32                     |                                           |
| Image Size          | 128×128                | Reduced from 256×256 for ~2× speed gain  |
| EarlyStopping       | `patience=5`, monitors `val_loss`, restores best weights |

---

## 🔄 Data Augmentation

Applied **only to training data** via `ImageDataGenerator`. Validation and test generators use only `preprocess_input` (no augmentation).

| Technique         | Value        | Reason                                                    |
|-------------------|--------------|-----------------------------------------------------------|
| Rotation          | ±180°        | X-rays can be taken at various orientations               |
| Zoom              | 25%          | Simulates varying image scales                            |
| Brightness        | [0.8, 1.2]   | Narrowed from [0.2, 0.5] — wider range obscured bone tip details |
| Height Shift      | 20%          | Simulates vertical positioning variance                   |
| Horizontal Flip   | ✅ Enabled   | Left/right hand symmetry                                  |
| Shear             | 0.05         | Minor geometric distortion                                |
| Fill Mode         | `nearest`    | Fills gaps after transformation                           |
| Preprocessing     | `preprocess_input` | Xception-specific normalization to [-1, 1] — **no manual rescale** |

> ⚠️ **Anti-Double Scaling:** `rescale=1/255` is intentionally **not used**. Xception's `preprocess_input` already normalizes pixel values to `[-1, 1]`. Applying both would corrupt the input distribution and degrade performance.

---

## 📊 Results

### Training History

| Epoch | Train Loss (MSE) | Train MAE | Val Loss (MSE) | Val MAE |
|-------|-----------------|-----------|----------------|---------|
| 1     | 7150.31         | 74.71     | 659.07         | 20.77   |
| 2     | 977.28          | 25.05     | 444.27         | 16.69   |
| 3     | 455.08          | 16.91     | 354.70         | 14.89   |
| 5     | 395.55          | 15.64     | 328.57         | 14.42   |
| 8     | 346.30          | 14.75     | **296.30**     | **13.37** |
| 11    | 313.63          | 13.90     | **277.54**     | **13.00** ← Best |
| 16    | 273.67          | 13.06     | 281.54         | 12.97   |

> EarlyStopping triggered at **epoch 16** (patience=5, best val_loss at epoch 11: **277.54 MSE**)  
> Best model restored automatically via `restore_best_weights=True`

### Figure 5 — Prediction Results on Test Set
![Prediction Results](Figure_5_Prediction_Results.png)

> Six test samples selected at equal intervals across the age spectrum (youngest to oldest). Each panel shows the hand X-ray alongside the real bone age and the model's predicted value (in months). Performance is strongest in the mid-range ages and weakest at the extremes — a common pattern in regression models trained on imbalanced age distributions.

### Final Performance (Best Checkpoint — Epoch 11)

| Metric           | Value              |
|------------------|--------------------|
| Best Val MSE     | **277.54**         |
| Best Val MAE     | **~13.0 months**   |

The model's average prediction error is approximately **±13 months**, which is within a clinically reasonable range for an automated baseline system on this dataset without gender input.

---

## ⚠️ Known Limitations

### 1. Gender Feature Not Used
The dataset includes gender information (`male`/`female`), which is a clinically significant factor in bone development. The current model uses only the X-ray image. Incorporating gender as an additional input (multi-input model) would likely improve accuracy meaningfully.

### 2. `norm_age` and `boneage_category` Prepared but Unused
These columns are computed during data preparation but are not passed to the model. They can be useful for analysis or as alternative training targets.

### 3. Small Regression Head
The classification head (`Dense(5) → Dense(1)`) is minimal. A larger head or additional dense layers may improve the model's ability to map Xception features to continuous age values.

### 4. Image Size Trade-off
Reducing from 256×256 to 128×128 nearly halved training time but discards fine-grained spatial detail (e.g., subtle growth plate features). Consider 256×256 with a GPU for better accuracy.

---

## 🚀 Setup & Usage

### Requirements

```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib pillow
```

### Dataset Structure

```
bonage_dataset/
├── boneage-training-dataset/
│   ├── 1377.png
│   ├── 1378.png
│   └── ...
├── boneage-test-dataset/
│   ├── 4360.png
│   └── ...
├── boneage-training-dataset.csv
└── boneage-test-dataset.csv
```

> ⚠️ After extracting the Kaggle ZIP, make sure PNG files sit **directly** inside `boneage-training-dataset/` and `boneage-test-dataset/` — not in a nested subfolder of the same name.

### Run

```bash
python main.py
```

---

## 🔧 Suggested Improvements

1. **Add gender as a second input** — Build a multi-input model that takes both the image and gender (0/1) as inputs. This is the single highest-impact improvement available in this dataset.

2. **Increase image resolution** — Return to 256×256 (or higher) with a GPU to recover spatial detail lost by downscaling.

3. **Larger regression head** — Replace `Dense(5)` with `Dense(256) → Dense(64) → Dense(1)` for richer feature mapping.

4. **Freeze then unfreeze (gradual fine-tuning)** — First train only the head with the base frozen, then gradually unfreeze Xception layers. This prevents the large initial loss spike (epoch 1: MSE 7150) caused by training all weights at once from the start.

5. **MAE as the primary loss** — MSE heavily penalizes outliers. Using MAE as the loss function may produce more stable and clinically interpretable results.

---

## 📁 Project Structure

```
📦 8-Kemik yasi tahmini/
├── main.py                    # Main training & evaluation script
├── README.md                  # This file
└── bonage_dataset/
    ├── boneage-training-dataset/
    ├── boneage-test-dataset/
    ├── boneage-training-dataset.csv
    └── boneage-test-dataset.csv
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Xception-red)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-train__test__split-green)
![License](https://img.shields.io/badge/Dataset_License-Kaggle_Competition-lightgrey)

---

> **Disclaimer:** This project is for educational purposes only and is not intended for clinical use.