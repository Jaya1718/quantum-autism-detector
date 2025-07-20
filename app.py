from flask import Flask, render_template, request
from preprocess import preprocess_data
from quantum_model import build_quantum_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    (X_train, X_test, y_train, y_test), _ = preprocess_data("AlzheimerDataset")
    model, acc = build_quantum_model(X_train, y_train, X_test, y_test)
    return f"Quantum model trained. Accuracy: {acc * 100:.2f}%"

if __name__ == '__main__':
    app.run(debug=True)
