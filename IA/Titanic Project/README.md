# Titanic Survival Prediction – Data Mining Pipeline

This repository presents a complete data mining pipeline using the famous Titanic dataset, aiming to explore and predict passenger survival based on demographic and socioeconomic features. The project combines Exploratory Data Analysis (EDA), classification models, clustering techniques, and association rule mining.

## 📁 Project Structure

- `Lista 11 - Pipeline e titanic.ipynb`: Jupyter Notebook with the full analysis and modeling.
- `titanic_dashboard_final.csv`: Final dataset used to generate the Looker Studio dashboard.
- `titanic_dashboard_link.txt`: Contains the link to the interactive dashboard on Looker Studio.
- `Predictive Survival Analysis – Titanic.pdf`: Downloaded dashboard from Looker Studio.

---

## 🔍 Pipeline Overview

### 1. Exploratory Data Analysis (EDA)

- Analyzed distributions of key variables such as `Sex`, `Pclass`, `Age`, `Fare`, and `Embarked`.
- Identified important survival patterns:
  - Females and 1st class passengers had the highest survival rates.
  - Female were also more likely to survive.
  - Higher ticket fares correlated with higher survival probabilities.

### 2. Classification Modeling

- Built supervised learning models to predict survival.
- Best model: **Random Forest**, achieving ~81.9% accuracy.
- Features used: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.
- Predictions were added to the dataset (`survived_pred`) and used in the dashboard.

### 3. Clustering (KMeans)

- Unsupervised learning with **KMeans** to segment passengers into behavioral clusters.
- Used attributes: `Pclass`, `Sex` (encoded), `Age`, `Fare`.
- 3 clusters were identified:
  - Cluster 0: Mostly 3rd class, low fare, low survival.
  - Cluster 1: Intermediate mix.
  - Cluster 2: Mostly 1st class, high fare, high survival.

### 4. Association Rule Mining (Apriori)

- Applied the **Apriori algorithm** to find frequent itemsets and association rules.
- Data was discretized and one-hot encoded.
- Example rules:
  - `{Sex=male, Pclass=3} ⇒ {Survived=0}` (Confidence: 0.84)
  - `{Sex=female, Pclass=1} ⇒ {Survived=1}` (Confidence: 0.91)
  - `{Age=0-10} ⇒ {Survived=1}` (Confidence: 0.85)

---

## 📊 Dashboard

An interactive dashboard was built using **Looker Studio**, allowing users to visually explore survival predictions and passenger profiles.

- Filters by age, class, gender, and predicted survival.
- Enables dynamic insights and storytelling from the Titanic dataset.

(https://lookerstudio.google.com/reporting/f3e03261-4bd0-47eb-938f-497b52164814)

---

## 🛠️ Technologies Used

- Python (Pandas, Scikit-learn, Mlxtend, Matplotlib/Seaborn)
- Jupyter Notebook
- Looker Studio (for dashboard visualization)

---

## ✅ Objectives Achieved

- Demonstrated a complete data mining pipeline from EDA to prediction and rule discovery.
- Built an interpretable dashboard for decision-making.
- Applied both supervised and unsupervised learning methods.

---

## 📌 Notes

This project was developed for academic purposes and showcases multiple techniques from the fields of data mining, machine learning, and data visualization.

---



