# 🧪 Toxic Comment Pipeline

This project focuses on building a machine learning pipeline for multi-label classification of toxic comments, using the **Toxic Comment Classification Challenge** dataset. The solution includes data preprocessing, model training and evaluation, and a visual dashboard for interpretability.

---

## 📌 Objective

The main goal is to develop and compare different machine learning models capable of identifying and classifying comments that contain:

- Toxic language
- Severe toxicity
- Obscene content
- Threats
- Insults
- Identity-based hate

The project uses a **multi-label classification** approach since a single comment can belong to multiple categories.

---

## 🛠️ Pipeline Overview

1. **Data Preprocessing**
   - Cleaning and vectorization using TF-IDF
   - Splitting of training and test sets

2. **Model Training**
   - Naive Bayes
   - Logistic Regression
   - Random Forest
   - All models use the One-vs-Rest strategy

3. **Evaluation**
   - Metrics used:
     - Accuracy
     - Hamming Loss
     - Average Precision
     - Average Recall
     - Average F1-score
   - Class-wise distribution and count comparison

4. **Visualization**
   - A Looker Studio dashboard was created to visualize model outputs and class distributions.

---

## 📊 Results Summary

| Model               | Accuracy | Hamming Loss | Avg Precision | Avg Recall | Avg F1 |
|---------------------|----------|--------------|----------------|------------|--------|
| Logistic Regression | 0.89     | 0.03         | 0.62           | 0.64       | 0.62   |
| Naive Bayes         | 0.90     | 0.03         | 0.71           | 0.50       | 0.58   |
| Random Forest       | 0.90     | 0.03         | 0.72           | 0.40       | 0.51   |

🔍 Logistic Regression achieved the best balance between precision and recall.  
📉 Random Forest had the highest precision but lowest recall, indicating under-detection of minority classes.  
⚖️ Naive Bayes showed solid performance but struggled with minority label detection.

---

## 📁 Repository Structure
Toxic_Comment_Pipeline/

├── class_counts_comparison.csv # Class distribution per model
├── classification_metrics.csv # Model performance metrics
├── Toxic_Comment_Analysis.pdf # Looker Studio dashboard
├── Toxic_Comment_Analysis.py # Full Python code for the ML pipeline


---

## 📈 Dashboard Insights

The visual dashboard presents:
- Model-wise prediction counts per class
- Class distribution (pie charts)
- Metrics comparison
- Helps identify class imbalance and model behavior on minority classes like `threat` and `identity_hate`
  (https://lookerstudio.google.com/reporting/b1baf6d1-4c36-44eb-96fc-43795be3f034) 

---

## 📚 Dataset

Source: [Kaggle - Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

---

## 

This project is for academic and experimental purposes.

---



