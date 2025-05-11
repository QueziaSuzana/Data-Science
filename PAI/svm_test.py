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

Xm_train, Xm_test, ym_train, ym_test = train_test_split(X, y, test_size=0.2, random_state=5, shuffle=False)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=5, shuffle=False)

binary_model = pickle.load(open("best_model_svm_binary.pkl", "rb"))
multiclass_model = pickle.load(open("best_model_svm_multiclass.pkl", "rb"))

y_pred_binary = binary_model.predict(Xb_train)
y_pred_multiclass = multiclass_model.predict(Xm_train)

print("\nBinary classification")
print("first 10 true:", np.array(list(yb_train[:10])))
print("first 10 predictions:", y_pred_binary[:10])
print()
print(confusion_matrix(yb_train, y_pred_binary))
print("Accuracy: ", binary_model.score(Xb_train, yb_train))

print("\nMulticlass classification")
print("first 10 true:", np.array(list(ym_train[:10])))
print("first 10:", y_pred_multiclass[:10])
print()
print(confusion_matrix(ym_train, y_pred_multiclass))
print("Accuracy: ", multiclass_model.score(Xm_train, ym_train))
