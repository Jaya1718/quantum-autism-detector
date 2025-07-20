from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit_aer import Aer

def build_quantum_model(X_train, y_train, X_test, y_test):
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1)
    ansatz = RealAmplitudes(num_qubits=X_train.shape[1], reps=1)
    optimizer = COBYLA(maxiter=100)

    backend = Aer.get_backend('aer_simulator')
    qi = QuantumInstance(backend)

    vqc = VQC(feature_map=feature_map,
              ansatz=ansatz,
              optimizer=optimizer,
              quantum_instance=qi)
    
    vqc.fit(X_train, y_train)
    score = vqc.score(X_test, y_test)
    return vqc, score
