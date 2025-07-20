import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

IMG_SIZE = 64

def load_data(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        path = os.path.join(data_dir, label)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                X.append(image)
                y.append(label)
            except Exception as e:
                continue
    return np.array(X), np.array(y)

def preprocess_data(data_dir):
    X, y = load_data(data_dir)
    X = X / 255.0
    X = X.reshape((X.shape[0], IMG_SIZE * IMG_SIZE))
    le = LabelEncoder()
    y = le.fit_transform(y)
    return train_test_split(X, y, test_size=0.2, random_state=42), le
