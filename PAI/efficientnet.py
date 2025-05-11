import os

import numpy as np
import pandas as pd

import pickle

import cv2

from sklearn import svm

import skimage
from skimage.io import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

filepath = "C:/Users/arthu/OneDrive/Área de Trabalho/puc/6Semester/PAI/TPClassifiers/TP - PAI"
#filepath = "C:/Users/Solides/Desktop/Codes/ClassificationImages/TP - PAI"

classes = ["Negative for intraepithelial lesion", "ASC-H", "ASC-US", "HSIL", "LSIL", "SCC"]

X_data = []
y_data = []

rotation_angles = [0, 90, 180, 270]

for i in classes:
    path = os.path.join(filepath, i)
    for img in os.listdir(path):
        flattened = imread(os.path.join(path, img))
        for angle in rotation_angles:
            rotated_img = cv2.rotate(flattened, cv2.ROTATE_90_CLOCKWISE * angle)
            rotated_img = cv2.resize(rotated_img, (224, 224))
            X_data.append(rotated_img)
            y_data.append(classes.index(i))

X = np.array(X_data)
y = np.array(y_data)

Xb = X.copy()
yb = y.copy()

for i in range(len(yb)):
    if yb[i] != 0:
        yb[i] = 1


#def downsampling_with_ranker(X, y):
#    ranker = svm.SVC(kernel='poly', degree=len(np.unique(y)), random_state=5, probability=True)
#    return ranker.fit(X, y).predict_proba(X).T

# get the n most likely to be in each class from the method downsampling_with_ranker

# mask1 and mask2 are the indices of the samples to be kept
# get the positions of the n most likely samples for each class

#dwr1 = downsampling_with_ranker(X, y)
#dwr2 = downsampling_with_ranker(Xb, yb)


#mask1 = np.hstack([np.argsort(dwr1[l])[ : 82 * 4] for l in np.unique(y)])  # 82: smallest class
#mask2 = np.hstack([np.argsort(dwr2[l])[ : 1337 * 4] for l in np.unique(yb)])  # 1337: classes 1, 2, 3, 4, 5 together

# multiply by 4 because of the 4 rotations (data augmentation)
# 82 and 1337 originally are the smallest class sizes. 82 for "SCC" and 1337 for all classes except "Negative for intraepithelial lesion"
mask1 = np.hstack([np.random.choice(np.where(y == l)[0], 82 * len(rotation_angles), replace=False) for l in np.unique(y)])  # 82: smallest class
mask2 = np.hstack([np.random.choice(np.where(yb == l)[0], 1337 * len(rotation_angles), replace=False) for l in np.unique(yb)])  # 1337: classes 1, 2, 3, 4, 5 together

print("Multiclass data information BEFORE")
print("before:", len(X))
print("class == 0:", len(X[y == 0]))
print("class == 1:", len(X[y == 1]))
print("class == 2:", len(X[y == 2]))
print("class == 3:", len(X[y == 3]))
print("class == 4:", len(X[y == 4]))
print("class == 5:", len(X[y == 5]))

print("Binary data information BEFORE")
print("before:", len(Xb))
print("class == 0:", len(Xb[yb == 0]))
print("class == 1:", len(Xb[yb == 1]))

assert mask1.max() < len(X)
assert mask2.max() < len(Xb)

X = X[mask1]
y = y[mask1]

Xb = Xb[mask2]
yb = yb[mask2]

print("Multiclass data information AFTER")
print("after:", len(X))
print("class == 0:", len(X[y == 0]))
print("class == 1:", len(X[y == 1]))
print("class == 2:", len(X[y == 2]))
print("class == 3:", len(X[y == 3]))
print("class == 4:", len(X[y == 4]))
print("class == 5:", len(X[y == 5]))
      
print("Binary data information AFTER")
print("after:", len(Xb))
print("class == 0:", len(Xb[yb == 0]))
print("class == 1:", len(Xb[yb == 1]))

# Configure GPU usage for TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')

hyperparameters = [
    {'weights': 'imagenet', 'include_top': False, 'batch_size': 64, 'epochs': 100, 'learning_rate': 0.00001, "dense": 512, "layers_amount": 2},
    # reduce more the learning rate: #{'weights': 'imagenet', 'include_top': False, 'batch_size': 64, 'epochs': 100, 'learning_rate': 0.000005, "dense": 512, "layers_amount": 2},
    # underfitting: #{'weights': 'imagenet', 'include_top': False, 'batch_size': 64, 'epochs': 100, 'learning_rate': 0.000005, "dense": 64, "layers_amount": 4},
]

# Binary classification

X_train, X_test, y_train, y_test = train_test_split(Xb, yb, test_size=0.2, random_state=5)

with open("output_efn_binary.txt", "w") as f:
    pass

for k in hyperparameters:
    base_model = EfficientNetB1(weights=k['weights'], include_top=k['include_top'], input_shape=(224, 224, 3))

    model = Sequential()
    model.add(base_model)
    
    model.add(GlobalAveragePooling2D())

    if k["dense"] > 0:
        for i in range(k["layers_amount"]):
            model.add(Dropout(0.5))
            model.add(Dense(k["dense"], activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))  # sigmoid is for binary classification

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=6)
    model_checkpoint = ModelCheckpoint(filepath='model_efn_binary.keras', monitor='val_accuracy', save_best_only=True)

    model.compile(optimizer=Adam(learning_rate=k['learning_rate']), loss='binary_crossentropy', metrics=['accuracy'])  # binary crossentropy is for binary classification
    
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=k["epochs"], batch_size=k["batch_size"], callbacks=[early_stopping, reduce_lr, model_checkpoint])

    with open("output_efn_binary.txt", "a") as f:
        f.write(f"Hyperparameters: {k}\n")

        predictions = model.predict(X_test)  # class probabilities
        predicted_classes = (predictions >= 0.5).astype(int)
        
        cm = confusion_matrix(y_test, predicted_classes)
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Confusion matrix (normalize true): \n{confusion_matrix(y_test, predicted_classes, normalize='true')}\n")
        f.write(f"--------------------------------------\n")
        print("One binary iteration finished!")

    print(f"Saving binary model and history for hyperparameters {k}...")

    model.save(f"model_efn_binary_{k['dense']}_{k['layers_amount']}.keras")
    pickle.dump(hist.history, open(f"hist_efn_binary_{k['dense']}_{k['layers_amount']}.pkl", "wb"))
    
print("DONE! All binary iterations finished!")

# Multiclass classification

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=5)

with open("output_efn_multiclass.txt", "w") as f:
    pass

for k in hyperparameters:
    base_model = EfficientNetB1(weights=k['weights'], include_top=k['include_top'], input_shape=(224, 224, 3))

    model = Sequential()
    model.add(base_model)

    model.add(GlobalAveragePooling2D())

    if k["dense"] > 0:
        for i in range(k["layers_amount"]):
            model.add(Dropout(0.5))
            model.add(Dense(k["dense"], activation='relu'))

    model.add(Dense(6, activation='softmax'))  # softmax is for multiclass classification

    model.compile(optimizer=Adam(learning_rate=k["learning_rate"]), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # sparse_categorical_crossentropy is for multiclass classification
    #model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # sparse_categorical_crossentropy is for multiclass classification

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4)

    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=k["epochs"], batch_size=k["batch_size"], callbacks=[early_stopping, reduce_lr])

    with open("output_efn_multiclass.txt", "a") as f:
        f.write(f"Hyperparameters: {k}\n")

        predictions = model.predict(X_test)  # class probabilities
        predicted_classes = np.argmax(predictions, axis=1)  # class indices

        cm = confusion_matrix(y_test, predicted_classes)
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Confusion matrix (normalize true): \n{confusion_matrix(y_test, predicted_classes, normalize='true')}\n")
        f.write(f"--------------------------------------\n")
        print("One multiclass iteration finished!")
    
    print(f"Saving multiclass model and history for hyperparameters {k}...")

    model.save(f"model_efn_multiclass_{k['dense']}_{k['layers_amount']}.keras")
    pickle.dump(hist.history, open(f"hist_efn_multiclass_{k['dense']}_{k['layers_amount']}.pkl", "wb"))
    pickle.dump(model, open(f"model_efn_multiclass_{k['dense']}_{k['layers_amount']}.pkl", "wb"))

print("DONE! All multiclass iterations finished!")