from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from preprocess import load_images

def train_quantum_model(dataset_path):
    X, y = load_images(dataset_path)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=1)
    ansatz = RealAmplitudes(num_qubits=X.shape[1], reps=1)

    quantum_instance = QuantumInstance(backend=AerSimulator())
    vqc = VQC(feature_map=feature_map,
              ansatz=ansatz,
              optimizer=COBYLA(maxiter=50),
              quantum_instance=quantum_instance)

    vqc.fit(X_train, y_train)
    y_pred = vqc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
