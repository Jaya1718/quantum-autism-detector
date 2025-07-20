from flask import Flask, render_template, request
import os
from autism_detector import predict_autism

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'mri_image' not in request.files:
        return render_template('index.html', result="No file uploaded.")

    file = request.files['mri_image']
    if file.filename == '':
        return render_template('index.html', result="No selected file.")

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Predict using VQC model
    result = predict_autism(filepath)

    return render_template('index.html', result=result, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
