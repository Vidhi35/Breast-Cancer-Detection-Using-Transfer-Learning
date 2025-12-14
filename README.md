# 🧬 Breast Cancer Detection Using Transfer Learning

This project implements a **Deep Learning-based Breast Cancer Detection System** using **Transfer Learning** on a **small medical image dataset**.  
The model classifies breast ultrasound images into **Benign**, **Malignant**, and **Normal** categories with **83.17% accuracy**.

---

## 📌 Project Overview

- **Domain**: Medical Image Analysis
- **Task**: Multi-class Image Classification
- **Classes**:
  - Benign
  - Malignant
  - Normal
- **Approach**: Transfer Learning
- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Framework**: TensorFlow / Keras
- **Platform**: Google Colab (GPU)
- **Accuracy**: 83.17%

---

## 📂 Dataset Information

**Dataset Name**: Breast Ultrasound Images Dataset (BUSI)

- Total Images: ~780
- Image Type: Ultrasound images
- Dataset Structure:

```
Dataset_BUSI_with_GT/
├── benign/
├── malignant/
└── normal/
```

- **Source**: [Kaggle - Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

---

## ⚙️ Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Keras

(Pre-installed on Google Colab)

---

## 🚀 Methodology

1. Load and preprocess ultrasound images
2. Apply data augmentation to reduce overfitting
3. Use **MobileNetV2** as a pretrained feature extractor
4. Add custom classification layers
5. Train the model on the BUSI dataset
6. Fine-tune the pretrained layers
7. Evaluate and test predictions on unseen images

---

## 🧠 Model Architecture

```
Input (224x224x3)
    ↓
MobileNetV2 (Pretrained, frozen base)
    ↓
Global Average Pooling 2D
    ↓
Dense Layer (128 units, ReLU activation)
    ↓
Dropout (0.5)
    ↓
Output Layer (3 units, Softmax activation)
```

**Key Components:**
- MobileNetV2 (without top layers)
- Global Average Pooling
- Dense (128 units, ReLU)
- Dropout (0.5)
- Dense Output Layer (Softmax – 3 classes)

---

## 📊 Training Details

- **Image Size**: 224 × 224
- **Batch Size**: 16
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**:
  - Initial Training: 10
  - Fine-Tuning: 5
- **Data Augmentation**: Rotation, flip, zoom, shift

---

## 📈 Performance Metrics

- **Test Accuracy**: 83.17%
- **Model Type**: Lightweight and efficient
- **Training Time**: Fast convergence due to transfer learning
- **Suitable For**: Small dataset scenarios

---

## 🧪 Testing

The trained model can predict the class of a single ultrasound image:
- **Benign**: Non-cancerous tumors
- **Malignant**: Cancerous tumors
- **Normal**: Healthy tissue

### Example Prediction:
```python
# Load and preprocess test image
test_image = load_and_preprocess_image('path/to/image.png')

# Make prediction
prediction = model.predict(test_image)
predicted_class = class_names[np.argmax(prediction)]

print(f"Predicted Class: {predicted_class}")
```

---

## 💾 Model Saving

The trained model is saved as:

```
breast_cancer_transfer_learning_model.h5
```

Can be loaded using:
```python
from tensorflow.keras.models import load_model
model = load_model('breast_cancer_transfer_learning_model.h5')
```

---

## 🎯 Key Features

- ✅ Transfer Learning with MobileNetV2
- ✅ Data Augmentation for better generalization
- ✅ Fine-tuning for improved accuracy
- ✅ Efficient training on small datasets
- ✅ 83.17% classification accuracy
- ✅ Lightweight model suitable for deployment

---

## 📌 Applications

- Medical image analysis
- Breast cancer screening assistance
- Academic and research projects
- AI-based healthcare systems
- Computer-aided diagnosis (CAD) tools

---

## 🚀 Getting Started

1. **Clone the repository**:
```bash
git clone https://github.com/Vidhi35/Breast-Cancer-Detection-Using-Transfer-Learning.git
cd Breast-Cancer-Detection-Using-Transfer-Learning
```

2. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

3. **Run the notebook** on Google Colab or Jupyter

4. **Train the model** or use the pretrained weights

5. **Test predictions** on new ultrasound images

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| **Accuracy** | 83.17% |
| **Model Size** | Lightweight (MobileNetV2) |
| **Training Time** | Fast (Transfer Learning) |
| **Classes** | 3 (Benign, Malignant, Normal) |

---

## ⚠️ Disclaimer

This project is intended **only for educational and research purposes**.  
It should **not** be used as a replacement for professional medical diagnosis.  
Always consult qualified healthcare professionals for medical decisions.

---

## 🙌 Author

**Developed by:** Vidhi Rani Netam  
**Field:** B.Tech (Honours) – Computer Science & Engineering (AI)  
**GitHub:** [@Vidhi35](https://github.com/Vidhi35)

---

## ⭐ Future Improvements

- [ ] Add Grad-CAM visualization for explainability
- [ ] Implement binary classification (Benign vs Malignant)
- [ ] Deploy as a web application using Flask/Streamlit
- [ ] Improve accuracy with larger datasets
- [ ] Integrate clinical metadata for better predictions
- [ ] Add ensemble learning with multiple models
- [ ] Implement cross-validation
- [ ] Create a mobile app version

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/Vidhi35/Breast-Cancer-Detection-Using-Transfer-Learning/issues).

---

## 📝 Citation

If you use this project in your research or work, please cite:

```
@misc{breast_cancer_detection_tl,
  author = {Vidhi Rani Netam},
  title = {Breast Cancer Detection Using Transfer Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Vidhi35/Breast-Cancer-Detection-Using-Transfer-Learning}
}
```

---

## 📎 License

This project is for academic use only.

---

## 🌟 Acknowledgments

- Dataset: [BUSI Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- Base Model: MobileNetV2 (TensorFlow/Keras)
- Platform: Google Colab

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
