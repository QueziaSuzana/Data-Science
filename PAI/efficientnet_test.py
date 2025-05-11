import pickle

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os

import cv2

from skimage.io import imread

import tensorflow as tf


np_array = np.array([[0.3110072 ],
 [0.5577462 ],
 [0.06806619],
 [0.4803013 ],
 [0.49906573],
 [0.0897857 ],
 [0.07762875],
 [0.10125897],
 [0.10222157],
 [0.4068583 ],
 [0.08035384],
 [0.14653102],
 [0.415443  ]])

print(np_array)
print(np_array[0])
print(np_array[1])
print(np_array.flatten())

print((np_array > 0.5).astype(int))
print(np_array.argmax(axis=1))

print("-------------------------------------------------")

print("Multi-class classification")

filepath = "C:/Users/arthu/OneDrive/Área de Trabalho/puc/6Semester/PAI/TPClassifiers/TP - PAI"
#filepath = "C:/Users/Solides/Desktop/Codes/ClassificationImages/TP - PAI"

classes = ["Negative for intraepithelial lesion", "ASC-H", "ASC-US", "HSIL", "LSIL", "SCC"]

X_data = []
y_data = []

rotation_angles = [0, 90, 180, 270]


model = tf.keras.models.load_model("model_efn_multiclass_512_2.h5")

for i in classes:
    path = os.path.join(filepath, i)
    image = imread(path)
    image = cv2.resize(image, (224, 224))
    X_data.append(image)

X_data = np.array(X_data)

predictions = model.predict(X_data)  # class probabilities

predicted_classes = predictions.argmax(axis=1)  # multiclass only
#predicted_classes = (predictions >= 0.5).astype(int)  # binary only

print(predicted_classes)


y = np.array(y_data)

Xb = X.copy()
yb = y.copy()

for i in range(len(yb)):
    if yb[i] != 0:
        yb[i] = 1

# multiply by 4 because of the 4 rotations (data augmentation)
# 82 and 1337 originally are the smallest class sizes. 82 for "SCC" and 1337 for all classes except "Negative for intraepithelial lesion"
mask1 = np.hstack([np.random.choice(np.where(y == l)[0], 82 * 4, replace=False) for l in np.unique(y)])  # 82: smallest class
mask2 = np.hstack([np.random.choice(np.where(yb == l)[0], 1337 * 4, replace=False) for l in np.unique(yb)])  # 1337: classes 1, 2, 3, 4, 5 together

assert mask1.max() < len(X)
assert mask2.max() < len(Xb)

X = X[mask1]
y = y[mask1]

Xb = Xb[mask2]
yb = yb[mask2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print("Original multiclass classification")

hist = pickle.load(open("hist_efn_multiclass_512_2.pkl", "rb"))
print(hist)
print(hist.keys())

model = tf.keras.models.load_model("model_efn_multiclass_512_2.h5")

original_multiclass_full_accuracy_sum = 0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # no random state

    predictions = model.predict(X_test)  # class probabilities
    predicted_classes = predictions.argmax(axis=1)
    print(predicted_classes)
    cm = confusion_matrix(y_test, predicted_classes)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #print(f"Multiclass accuracy: {accuracy}")
    #print(f"Multiclass confusion matrix: \n{cm}")

    original_multiclass_full_accuracy_sum += accuracy

print(f"Average accuracy in 10 runs: {original_multiclass_full_accuracy_sum / 10}")

print("-------------------------------------------------")
print("'Binarized' multiclass classification")


binarized_multiclass_full_accuracy_sum = 0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # no random state

    binarized_y = (y_test > 0).astype(int)
    
    predictions = model.predict(X_test)  # class probabilities
    predicted_classes_binarized = (predictions.argmax(axis=1) > 0).astype(int)

    cm = confusion_matrix(binarized_y, predicted_classes_binarized)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    #print(f"Multiclass accuracy: {accuracy}")
    #print(f"Multiclass confusion matrix: \n{cm}")

    binarized_multiclass_full_accuracy_sum += accuracy

print(f"Average accuracy in 10 runs: {binarized_multiclass_full_accuracy_sum / 10}")

print("-------------------------------------------------")
print("Multiclass classification with pickle model")

model = pickle.load(open("model_efn_multiclass_512_2.pkl", "rb"))

predictions = model.predict(X_test)  # class probabilities
predicted_classes = predictions.argmax(axis=1)
        
cm = confusion_matrix(y_test, predicted_classes)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

print(f"Multiclass accuracy: {accuracy}")
print(f"Multiclass confusion matrix: \n{cm}")

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("-------------------------------------------------")
print("Binary classification")

#hist = pickle.load(open("hist_efn_binary_8_31.pkl", "rb"))
#print(hist)
#print(hist.keys())

X_train, X_test, y_train, y_test = train_test_split(Xb, yb, test_size=0.2, random_state=5)

model = tf.keras.models.load_model("model_efn_binary_8_3.pkl")

predictions = model.predict(X_test)  # class probabilities
predicted_classes = (predictions > 0.5).astype(int)

cm = confusion_matrix(y_test, predicted_classes)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

print(f"binary accuracy: {accuracy}")
print(f"binary confusion matrix: \n{cm}")

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
