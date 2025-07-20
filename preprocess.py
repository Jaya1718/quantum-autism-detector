import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def load_images(dataset_path, img_size=(64, 64), max_images_per_class=100):
    X, y = [], []
    labels = os.listdir(dataset_path)

    for label in labels:
        class_dir = os.path.join(dataset_path, label)
        count = 0
        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path).convert('L')  # grayscale
                img = img.resize(img_size)
                X.append(np.array(img).flatten() / 255.0)
                y.append(label)
                count += 1
                if count >= max_images_per_class:
                    break

    X = np.array(X)
    y = LabelEncoder().fit_transform(y)
    return X, y
