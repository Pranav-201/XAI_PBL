# 🧠 Pneumonia Detection using DenseNet121 + Grad-CAM (XAI)

## 📘 Overview
This project implements a deep learning pipeline for **pneumonia detection** using **Chest X-ray images**.  
The model is trained on the publicly available **Kaggle Chest X-Ray Pneumonia Dataset** and enhanced with **Grad-CAM** visualization to explain the model’s predictions.

It uses **PyTorch**, **DenseNet121**, and **Explainable AI** techniques to highlight regions in the X-ray that contributed most to the model’s decision.

---

## 🧩 Features
- Automated dataset download from Kaggle  
- Data preprocessing and augmentation using `torchvision.transforms`  
- Transfer learning using `DenseNet121`  
- Model training and evaluation with ROC-AUC scoring  
- Grad-CAM visualization for explainable AI insights  
- Inference on new uploaded chest X-ray images  

---

## ⚙️ Requirements
Make sure the following libraries are installed:

```bash
pip install torch torchvision scikit-learn opencv-python tqdm matplotlib kaggle pillow numpy
```

If you want to train on GPU, ensure **CUDA** is available.

---

## 📂 Dataset
Dataset used: [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Folder structure after extraction:

```
chest_xray/
├── train/
│   ├── PNEUMONIA/
│   └── NORMAL/
├── val/
│   ├── PNEUMONIA/
│   └── NORMAL/
└── test/
    ├── PNEUMONIA/
    └── NORMAL/
```

---

## 🚀 How to Run (on VS Code)
1. **Clone or download this repo**  
   ```bash
   git clone https://github.com/yourusername/pneumonia-xai.git
   cd pneumonia-xai
   ```

2. **Ensure the dataset is available locally**  
   You can manually download from Kaggle and extract it to the project folder as shown above.

3. **Run the Python script**  
   ```bash
   python XAI.py
   ```

4. **Upload a new image for testing**  
   After training, the script will prompt you to select a new X-ray image.  
   The Grad-CAM heatmap will be displayed to explain the model’s prediction.

---

## 🧮 Model Details
- **Architecture:** DenseNet121 (Pretrained on ImageNet)  
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam (LR = 1e-4)  
- **Evaluation Metric:** ROC-AUC  

---

## 🔍 Explainable AI (Grad-CAM)
Grad-CAM highlights the important regions of the input X-ray image that influenced the model’s classification decision.  
This helps doctors and researchers **visualize what the model “sees”** as pneumonia symptoms.

Generated output example:

| Original X-ray | Grad-CAM Heatmap | Overlay |
|----------------|------------------|----------|
| ![original](assets/original.jpg) | ![gradcam](assets/gradcam.jpg) | ![overlay](assets/overlay.jpg) |

---

## 🏆 Results
- Achieved strong validation performance with ROC-AUC > 0.95  
- Clear Grad-CAM visualization showing pneumonia-affected lung regions  

---

## 📚 References
- [Kaggle: Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---


