# Automated-Early-Diabetic-Retinopathy-Detection-Using-a-Deep-Hybrid-CNN-Model
Implementation of a deep hybrid CNN (VGG16 + InceptionV3) for automated diabetic retinopathy detection on the EyePACS dataset. Includes bicubic preprocessing, geometric data augmentation, and two-stage transfer learning.

# 🩺 Diabetic Retinopathy Detection Using a Deep Hybrid CNN (VGG16 + InceptionV3)

A deep learning project for **automated diabetic retinopathy (DR) detection** using a hybrid CNN combining **VGG16** and **InceptionV3** architectures.  
Trained and evaluated on the **EyePACS** retinal fundus image dataset (~35K images).

---

## 📘 Project Overview
This project aims to automatically classify retinal fundus images into **five diabetic retinopathy (DR) severity levels**:
- 0 → No DR  
- 1 → Mild  
- 2 → Moderate  
- 3 → Severe  
- 4 → Proliferative DR  

The model uses transfer learning with pretrained **VGG16** and **InceptionV3** backbones, fusing both feature maps to achieve improved representation learning.

---

## 🧠 Dataset
- **Dataset:** EyePACS (Kaggle Diabetic Retinopathy Detection Challenge)  
- **Total images:** 35,126  
- **Valid images after cleaning:** 35,122 (4 corrupted removed)  
- **Classes:** 0–4  
- **Split:** 80% Train / 10% Validation / 10% Test  
- **CSV files generated:**  
  - `eyepacs_all_fixed.csv` → Cleaned dataset  
  - `eyepacs_split.csv` → Train/Val/Test split  

### Class Distribution
<img width="665" height="603" alt="image" src="https://github.com/user-attachments/assets/1c819525-d7be-49f2-8b19-e184a8011edf" />


---

## ⚙️ Preprocessing Pipeline
- Resized images using **bicubic interpolation** → 299×299×3  
- Normalized pixel values to [0, 1]  
- Removed corrupted or unreadable files  
- No extra preprocessing (no CLAHE, gamma, or cropping

---

## 🔁 Data Augmentation
To increase data diversity and reduce overfitting, the following augmentations were applied during training:
| Augmentation | Range / Description |
|---------------|---------------------|
| Horizontal Flip | Randomly flips images left–right |
| Rotation | ±15° |
| Zoom | ±10 % |
| Crop | Random crop to 299×299 |
| Shear | ±10° affine shear |

### Example – Before vs After Augmentation
![Augmentation Example]

---

## 🧩 Model Architecture — Hybrid CNN

| Component | Description |
|------------|-------------|
| **Backbones** | VGG16 + InceptionV3 (ImageNet pretrained) |
| **Feature Extraction** | VGG16 `block5_conv3`, InceptionV3 `mixed10` |
| **Pooling** | Global Average Pooling |
| **Fusion** | Concatenation of both pooled features |
| **Fully Connected Layers** | Dense(4096) → Dropout(0.5) → Dense(4096) → Dropout(0.5) |
| **Output Layer** | Dense(5, Softmax) |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Optimizer** | Adam (lr = 1e-4 → 1e-5 for fine-tuning) |
| **Batch Size** | 32 |

---

## 🏋️ Training Strategy
Two-stage transfer learning:
1. **Stage 1 – Feature Extraction:**  
   - Freeze both base models  
   - Train only the fully connected head (10 epochs)

2. **Stage 2 – Fine-Tuning:**  
   - Unfreeze top 30 layers of both backbones  
   - Train entire hybrid model at a lower learning rate (10 epochs)  
   - EarlyStopping, ReduceLROnPlateau, ModelCheckpoint used  

---

## 📊 Results

| Metric | Score |
|:--------|:------:|
| **Test Accuracy** | **0.8098** |
| **Macro Precision** | 0.5690 |
| **Macro Recall** | 0.4678 |
| **Macro F1-score** | 0.4897 |
| **Weighted F1-score** | 0.7714 |



---

## 📈 Training Curves
<img width="1370" height="563" alt="Screenshot 2025-10-23 173445" src="https://github.com/user-attachments/assets/f077ad17-bc9c-4fc1-aa2a-ba6671d62075" />


 
## 🧾 Key Files

| File | Description |
|------|--------------|
| **`main.ipynb`** | Complete Google Colab notebook containing the full preprocessing, training, and evaluation code |
| **`main.pdf`** | Detailed project report with explanations, architecture diagrams, and results |
| **`eyepac.csv`** | Original dataset CSV used for mapping image filenames to labels |
| **`requirements.txt`** | Python dependencies for reproducing the project |
| **Dataset Link** | [EyePACS on Kaggle](https://www.kaggle.com/competitions/diabetic-retinopathy-detection) |
