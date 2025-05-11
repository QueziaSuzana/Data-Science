import numpy as np
import pandas as pd

import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

filepath = "C:/Users/arthu/OneDrive/Área de Trabalho/puc/6Semester/PAI/TPClassifiers/haralick_features.csv"
filepath_binary = "C:/Users/arthu/OneDrive/Área de Trabalho/puc/6Semester/PAI/TPClassifiers/haralick_features_binary.csv"
#filepath = "C:/Users/Solides/Desktop/Codes/ClassificationImages/haralick_descriptors.csv"

df = pd.read_csv(filepath, sep=",")
dfb = pd.read_csv(filepath_binary, sep=",")

X = df.drop("target", axis=1)
y = df["target"]

Xb = dfb.drop("target", axis=1)
yb = dfb["target"]

hyperparameters = [
    {'kernel': 'rbf', 'C': 0.1, "degree": 0},
    {'kernel': 'rbf', 'C': 1, "degree": 0},
    {'kernel': 'rbf', 'C': 10, "degree": 0},

    {'kernel': 'poly', 'C': 0.1, "degree": 1},
    {'kernel': 'poly', 'C': 1, "degree": 1},
    {'kernel': 'poly', 'C': 10, "degree": 1},

    {'kernel': 'poly', 'C': 0.1, "degree": 2},
    {'kernel': 'poly', 'C': 1, "degree": 2},
    {'kernel': 'poly', 'C': 10, "degree": 2},
    
    {'kernel': 'poly', 'C': 0.1, "degree": 3},
    {'kernel': 'poly', 'C': 1, "degree": 3},
    {'kernel': 'poly', 'C': 10, "degree": 3},

    {'kernel': 'poly', 'C': 0.1, "degree": 4},
    {'kernel': 'poly', 'C': 1, "degree": 4},
    {'kernel': 'poly', 'C': 10, "degree": 4},

    {'kernel': 'poly', 'C': 0.1, "degree": 5},
    {'kernel': 'poly', 'C': 1, "degree": 5},
    {'kernel': 'poly', 'C': 10, "degree": 5},

    {'kernel': 'poly', 'C': 0.1, "degree": 6},
    {'kernel': 'poly', 'C': 1, "degree": 6},
    {'kernel': 'poly', 'C': 10, "degree": 6},

    {'kernel': 'poly', 'C': 0.1, "degree": 7},
    {'kernel': 'poly', 'C': 1, "degree": 7},
    {'kernel': 'poly', 'C': 10, "degree": 7},

]

# Binary classification

X_train, X_test, y_train, y_test = train_test_split(Xb, yb, test_size=0.2, random_state=5)

with open("output_svm_binary.txt", "w") as f:
    pass

highest_accuracy = 0
best_model = None

for k in hyperparameters:
    model = SVC(kernel=k['kernel'], C=k['C'], degree=k['degree'])
    model.fit(Xb, yb)

    accuracy = model.score(X_test, y_test)
    with open("output_svm_binary.txt", "a") as f:
        f.write(f"Hyperparameters: {k}\n")
        f.write(f"Full class instance: {model}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Confusion matrix: \n{confusion_matrix(y_test, model.predict(X_test), normalize='all')}\n")
        f.write(f"--------------------------------------\n")
        print("One binary iteration finished!")
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_model = model
print("DONE! All binary iterations finished!")

print("-------------------------------------------------")
print("best binary accuracy: ", highest_accuracy)
print("best binary model: ", best_model)

with open("best_model_svm_binary.pkl", "wb") as f:
    pickle.dump(best_model, open("nome_modelo.pkl", "wb"))

# Multiclass classification

with open("output_svm_multiclass.txt", "w") as f:
    pass

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

highest_accuracy = 0
best_model = None
for k in hyperparameters:
    model = SVC(kernel=k['kernel'], C=k['C'], degree=k['degree'])
    model.fit(X, y)

    accuracy = model.score(X_test, y_test)
    best_model = model
    with open("output_svm_multiclass.txt", "a") as f:
        f.write(f"Hyperparameters: {k}\n")
        f.write(f"Full class instance: {model}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Confusion matrix: \n{confusion_matrix(y_test, model.predict(X_test), normalize='all')}\n")
        f.write(f"--------------------------------------\n")
        print("One multiclass iteration finished!")
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_model = model
print("DONE! All multiclass iterations finished!")
print("-------------------------------------------------")
print("best multiclass accuracy: ", highest_accuracy)
print("best multiclass model: ", best_model)


with open("best_model_svm_multiclass.pkl", "wb") as f:
    pickle.dump(best_model, f)
    