# Cervical Cancer Detection with Pap Smear Images

This project explores the use of both shallow and deep learning techniques for the classification of cervical cancer using Pap smear images.

## 📌 About the Project

The goal of this study is to compare the performance of traditional machine learning (SVM with Haralick texture features) and deep learning (EfficientNetB1) approaches in classifying cell images from Pap smear exams into binary (cancer vs. non-cancer) and multi-class categories (6 diagnostic classes).

The experiments were conducted on a dataset of 5,581 images, with classes ranging from “Negative for intraepithelial lesion” to “SCC (Squamous Cell Carcinoma).” Data augmentation was applied by rotating each image at 0°, 90°, 180°, and 270°, increasing the dataset size fourfold.

## 🧪 Methodology

- **Shallow Classifier (SVM)**: Used Haralick texture descriptors (entropy, homogeneity, and contrast) extracted from six gray-level co-occurrence matrices (C1,1 to C32,32).
- **Deep Classifier (EfficientNetB1)**: Used raw 100x100 RGB images as input with extensive hyperparameter tuning.
- **Image Processing**: Performed using OpenCV and Pillow; descriptors calculated with NumPy.
- **GUI**: A simple interface was built using Tkinter to demonstrate results.

## 📈 Results

The results can be explored in the articles: papaer_english and paper_portuguese

## 🧾 Reference

This work is based on the paper:

> Arthur S. Quadros, Quézia P. Silva, Sarah S. Magalhães. *Cervical Cancer Detection with the Papanicolaou Smear with Shallow and Deep Classifiers*. PUC Minas.

---

For any questions or suggestions, please reach out!
