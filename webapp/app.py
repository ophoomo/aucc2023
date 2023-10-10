from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['RESULT_FOLDER']):
    os.makedirs(app.config['RESULT_FOLDER'])

# Load the pre-trained model
model = tf.keras.models.load_model('CADDetection.h5')
class_names = ['BED-DOUBLE', 'BED-SINGLE', 'DOOR-DOUBLE', 'DOOR-SINGLE', 'DOOR-WINDOWED', 'SHOWER', 'SINK',
                'SOFA-CORNER', 'SOFA-ONE', 'SOFA-THREE', 'SOFA-TWO', 'TABLE-DINNER', 'TOILET', 'WASHBASIN', 'WINDOW']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        
        if not file:
            flash('Please select a file')
            return redirect(request.url)
        
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (250, 250))
        image = np.reshape(image, (1, 250, 250, 1))
        
        if image is not None:
            predictions = model.predict(image)
            class_name = class_names[np.argmax(predictions[0])]
            num = predictions[0][np.argmax(predictions[0])]*100
        
        result_filename = os.path.join(app.config['RESULT_FOLDER'], file.filename)
        cv2.imwrite(result_filename, image)

        url = url_for('result', filename=file.filename)

        return render_template('index.html', image=url, name=class_name, num=num)

    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)