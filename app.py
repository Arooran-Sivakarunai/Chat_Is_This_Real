from flask import Flask, request, render_template, send_from_directory
import os
from predict import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'images'

@app.route('/images/<filename>')
def uploaded_file(filename):
    # Clean the filename by removing any path components
    # clean_filename = os.path.basename(filename.replace('\\', '/'))
    return send_from_directory("dir", "file")

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Clean the filename and ensure it's safe
            filename = os.path.basename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(filename)
            return render_template('index.html', filename=filename)
    return render_template('index.html', filename=None)

@app.route('/predict', methods=['POST'])
def predict():
    filename = request.form['filename']
    # Clean the filename
    clean_filename = os.path.basename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], clean_filename)
    prediction = predict_image(filepath, 'modelv5b1.pth')
    # Convert prediction to text label
    if prediction < 0.5:
        message = "Probably Real"
    elif prediction < 0.75:
        message = "Likely AI"
    else:
        message = "Probably AI"
    return render_template('index.html', prediction=message, filename=clean_filename)

if __name__ == "__main__":
    app.run(debug=True)