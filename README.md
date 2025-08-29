# 🚀 From Scratch to Smart: CNNs vs Transfer Learning on CIFAR-10

<p align="center">
  <img src="https://github.com/sergie-o/cnn-vs-transfer-cifar10/blob/main/image.png" alt="Banner" width="900"/>
</p>  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)  
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)  
![Keras](https://img.shields.io/badge/API-Keras-red.svg)  
![Dataset](https://img.shields.io/badge/Data-CIFAR--10-blueviolet.svg)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)  


> 🧠 **What happens when you compare custom-built CNNs against modern transfer learning models?**  
> This project explores exactly that by benchmarking **CNN architectures vs MobileNetV2 & EfficientNetB0** on the **CIFAR-10 dataset**.  

Unlike many CIFAR-10 projects that focus only on one approach, this study walks through the **evolution of improvements**:  
- Starting from a **basic CNN built from scratch**  
- Adding **early stopping and data augmentation** to reduce overfitting  
- Scaling up to a **regularized CNN with BatchNorm & Dropout**  
- Finally, benchmarking against **transfer learning with pretrained ImageNet models**  

The result is a **clear, data-driven progression** that shows how each enhancement contributes — from ~67% accuracy with the first CNN to **92.7% with EfficientNetB0**.   

---

## 📌 Overview  
Most CIFAR-10 projects stop at building a simple CNN.  
This project reframes the dataset as a **controlled experiment** to explore:  

- How much performance can be achieved by tuning **handcrafted CNNs**?  
- When does **transfer learning** clearly dominate?  
- Which architectural improvements (BatchNorm, L2, Dropout) bridge the gap?  

The final **EfficientNetB0** model achieved **92.7% accuracy**, outperforming all CNN baselines.  

---

## 📂 Dataset Description  
- **Source**: CIFAR-10 (via `tensorflow.keras.datasets`)  
- **Size**:  
  - 50,000 training images  
  - 10,000 test images  
- **Classes**: 10 categories → airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
- **Preprocessing**:  
  - Normalization (`/255` or Rescaling layer)  
  - One-hot encoding for categorical crossentropy  
  - Resized inputs for transfer learning (`96×96` for MobileNetV2, `224×224` for EfficientNetB0)  

---

## 🎯 Research Goals  
1. Benchmark **baseline CNN** performance  
2. Incrementally improve CNNs with:  
   - Early stopping  
   - Data augmentation  
   - Batch normalization  
   - Regularization (L2, Dropout)  
3. Compare CNNs against **transfer learning models**  
4. Evaluate trade-offs: training cost vs generalization vs accuracy  

---
## 🛠 Steps Taken  

The project followed a **progressive benchmarking approach**, starting with simple CNNs and gradually moving towards advanced transfer learning models.  

1. **Baseline CNN (Model 1)**  
   - Architecture: Conv → Pool → Dense  
   - Result: **67.6% accuracy**  

2. **CNN + EarlyStopping**  
   - Added `EarlyStopping(patience=3)` to prevent overfitting  
   - Result: **69.4% accuracy**  

3. **CNN + Data Augmentation + EarlyStopping**  
   - Applied random rotations, shifts, flips for better generalization  
   - Result: **74.1% accuracy**  

4. **Stronger CNN (Model 2)**  
   - Added **Batch Normalization**, **L2 regularization**, and **Dropout**  
   - Result: **84.2% accuracy**  

5. **Transfer Learning – MobileNetV2**  
   - Pretrained on ImageNet, frozen base, custom classification head  
   - Result: **85.9% accuracy**  

6. **Transfer Learning – EfficientNetB0**  
   - Pretrained on ImageNet, fine-tuned last ~30 layers  
   - Result: **92.7% accuracy (best model)**  

---  
📈 This stepwise progression shows how **each improvement contributed** — from architectural tweaks in CNNs to the power of pretrained models.  
## 📊 Key Findings  

| Model Variant                       | Key Improvements                     | Test Accuracy |
|-------------------------------------|--------------------------------------|---------------|
| **CNN v1 – Baseline**               | Basic Conv → Pool → Dense            | **0.6765** |
| **CNN v1 + EarlyStopping**          | Overfitting prevention               | **0.6946** |
| **CNN v1 + Data Aug + ES**          | Data diversity + early stopping      | **0.7415** |
| **CNN v2 – Stronger CNN**           | BN + L2 + Dropout regularization     | **0.8426** |
| **Transfer Learning – MobileNetV2** | Pretrained @96×96 + fine-tuned head  | **0.8588** |
| **Transfer Learning – EfficientNetB0** | Pretrained @224×224 + fine-tuning | **0.9267** |

✅ **Insight**: Transfer learning dominates, but augmentation + regularization dramatically improved CNN performance before hitting the transfer learning ceiling.  

---

## 📈 Accuracy Progression  

<p align="center">
  <img src="https://github.com/sergie-o/cnn-vs-transfer-cifar10/blob/main/%F0%9F%93%8A%20Accuracy%20Progression%3A%20CNN%20vs%20Transfer%20Learning.png" alt="Accuracy Progression Chart" width="700"/>
</p>  

---

## 🔍 Confusion Matrix Insights  

### CNN v1 (Baseline)  
- Confusion between **cat vs dog**, **truck vs automobile**  
- Recall weak for animals  

### CNN v1 + Data Augmentation  
- Reduced confusion in vehicle classes  
- Better recall for cats/dogs  

### Stronger CNN (BN + L2 + Dropout)  
- Much cleaner separation across most classes  
- Test accuracy jumped to **84%**  

### EfficientNetB0 (Best Model)  
- Very few cross-class errors  
- Clear distinction between animals & vehicles  
- Best generalization (**92.7%**)  


---

## 💻 Reproduction Guide  

**Requirements**:  
- `tensorflow`  
- `keras`  
- `numpy`, `pandas`  
- `scikit-learn`  
- `matplotlib`, `seaborn`, `plotly`


### 1️⃣ Install dependencies  
```bash
pip install -r requirements.txt
# 2) Run analysis & plots (VS Code)
jupyter notebook notebooks/computervsion.ipynb

# 3) Train CNNs (Colab)
#    - Model 1: Baseline CNN
#    - Model 1 + EarlyStopping
#    - Model 1 + Data Augmentation + EarlyStopping
#    - Model 2: Stronger CNN (BN, L2, Dropout)
open notebooks/cnn_models.ipynb

# 4) Train Transfer Learning models (Colab)
#    - MobileNetV2 (96×96, frozen base → fine-tune)
#    - EfficientNetB0 (224×224, frozen base → fine-tune last ~30 layers)
open notebooks/transfer_models.ipynb

# 5) Results & reports
# All evaluation, confusion matrices, and summaries 
# are included in computervsion.ipynb

---
```
📁 Repo Structure

```
cifar10-vision-benchmark/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt                
├─ notebooks/
│  ├─ computervsion.ipynb       # VS Code: data exploration + reports + evaluation
│  ├─ cnn_models.ipynb          # Colab: CNN experiments (baseline → aug → stronger CNN)
│  └─ transfer_models.ipynb     # Colab: Transfer learning (MobileNetV2, EfficientNetB0)
├─ models/                      # Saved models (.keras, .h5) – ignored by git
│  └─ .gitkeep
├─ reports/
│  ├─ figures/
│  │  ├─ cm_cnn_baseline.png
│  │  ├─ cm_cnn_augmented.png
│  │  ├─ cm_cnn_stronger.png
│  │  ├─ cm_mobilenetv2.png
│  │  └─ cm_efficientnetb0.png
│  └─ feature/
│     ├─ linkedin_feature.png
│     └─ summary_infographic.png
└─ logs/
   └─ training_histories/       # JSON histories per run (if saved)
