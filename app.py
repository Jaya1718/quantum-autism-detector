from flask import Flask, render_template, request
from quantum_model import train_quantum_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    accuracy = train_quantum_model("AlzheimerDataset")
    return render_template("index.html", accuracy=round(accuracy * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
