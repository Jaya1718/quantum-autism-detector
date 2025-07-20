import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from qiskit_aer import Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import Estimator
from PIL import Image
import nibabel as nib
import joblib  # For saving and loading model

# Paths
dataset_path = "Autism"
model_data_path = "vqc_model_data.npz"
vqc_model_path = "vqc_trained_model.pkl"

def load_mri_data(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.nii'):
            filepath = os.path.join(directory, filename)
            image = nib.load(filepath).get_fdata()
            image = np.mean(image, axis=2)  # Convert 3D to 2D
            image = Image.fromarray(image).resize((64, 64))  # Resize
            image_array = np.array(image).flatten()  # Flatten
            data.append(image_array)
    return data

def prepare_data():
    autistic_dir = os.path.join(dataset_path, 'autistic')
    non_autistic_dir = os.path.join(dataset_path, 'non-autistic')

    X_autistic = load_mri_data(autistic_dir)
    X_non_autistic = load_mri_data(non_autistic_dir)

    X = X_autistic + X_non_autistic
    y = ['autistic'] * len(X_autistic) + ['non-autistic'] * len(X_non_autistic)

    if not X:
        raise ValueError("Dataset is empty. Please check your image paths and preprocessing.")

    pca = PCA(n_components=4)
    X_reduced = pca.fit_transform(np.array(X))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return train_test_split(X_reduced, y_encoded, test_size=0.2, random_state=42), pca, label_encoder

def build_vqc():
    feature_map = ZZFeatureMap(4)
    ansatz = RealAmplitudes(4, reps=1)
    backend = Aer.get_backend('aer_simulator')
    estimator = Estimator()
    return VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=100), estimator=estimator)

def train_and_save_vqc():
    np.random.seed(42)
    (X_train, X_test, y_train, y_test), pca, label_encoder = prepare_data()

    vqc = build_vqc()
    vqc.fit(X_train, y_train)

    # Save VQC model
    joblib.dump(vqc, vqc_model_path)

    # Save PCA and label encoder data
    np.savez(model_data_path,
             components=pca.components_,
             mean=pca.mean_,
             classes=label_encoder.classes_)

def predict_autism(image_path):
    if not os.path.exists(vqc_model_path) or not os.path.exists(model_data_path):
        train_and_save_vqc()

    try:
        image = nib.load(image_path).get_fdata()
        image = np.mean(image, axis=2)
        image = Image.fromarray(image).resize((64, 64))
        image_array = np.array(image).flatten()
    except Exception as e:
        return "❌ Error loading .nii file: " + str(e)

    # Load PCA and label encoder info
    model_data = np.load(model_data_path, allow_pickle=True)
    components = model_data['components']
    mean = model_data['mean']
    classes = model_data['classes']

    pca = PCA(n_components=4)
    pca.components_ = components
    pca.mean_ = mean

    label_encoder = LabelEncoder()
    label_encoder.classes_ = classes

    # Transform input
    X_input = pca.transform([image_array])

    # Load trained model
    vqc = joblib.load(vqc_model_path)

    # Predict
    prediction = vqc.predict(X_input)
    return label_encoder.inverse_transform(prediction)[0]
