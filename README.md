# 📌 Benchmarking CNNs vs Transfer Learning — CIFAR-10 Case Study  

<p align="center">
  <img src="" alt="Accuracy Progression Chart" width="700"/>
</p>  


![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)  
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)  
![Keras](https://img.shields.io/badge/API-Keras-red.svg)  
![Dataset](https://img.shields.io/badge/Data-CIFAR--10-blueviolet.svg)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)  

> 🧠 **How far can we push CNNs, and when does transfer learning outperform them?**  
> This project benchmarks **custom CNNs** against **pretrained models (MobileNetV2, EfficientNetB0)** on the CIFAR-10 dataset, progressively improving models with **early stopping, augmentation, regularization, and fine-tuning**.  

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

1. **Baseline CNN (Model 1)**  
   - Conv → Pool → Dense  
   - Accuracy: **67.6%**  

2. **CNN + EarlyStopping**  
   - Added `EarlyStopping(patience=3)`  
   - Accuracy: **69.4%**  

3. **CNN + Augmentation + EarlyStopping**  
   - Applied random rotations, shifts, flips  
   - Accuracy: **74.1%**  

4. **Stronger CNN (Model 2)**  
   - Added **BatchNorm**, **L2 regularization**, **Dropout**  
   - Accuracy: **84.2%**  

5. **Transfer Learning – MobileNetV2**  
   - Pretrained on ImageNet, frozen base  
   - Accuracy: **85.9%**  

6. **Transfer Learning – EfficientNetB0**  
   - Pretrained on ImageNet, fine-tuned last ~30 layers  
   - Accuracy: **92.7%**  

---

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

**Steps**:  
```bash
git clone https://github.com/<your-username>/cifar10-vision-benchmark.git
cd cifar10-vision-benchmark
pip install -r requirements.txt
