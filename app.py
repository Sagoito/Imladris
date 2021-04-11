from skimage import io

from src.data_preprocesing import prepare_data
from src.mrcnn import visualize
from src.settings import Setting
import os
from pathlib import Path

import matplotlib.pyplot as plt
import src.coco as network
import threading

from flask import Flask
from flask import render_template, request, flash, redirect, session, url_for
from werkzeug.utils import secure_filename


app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(setting.user_image, filename))
            session['filename'] = filename
            return redirect(url_for('show_image'))

    return render_template('index.html')


def segment_image(filename, model):
    image = io.imread(filename)
    # # Run detection
    results = model.detect([image], verbose=1)
    #
    # # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])



@app.route('/show_image', methods=['GET', 'POST'])
def show_image():
    filename = session['filename']
    file_path = f'user_image/{filename}'
    return render_template('show_image.html', name=file_path)


if __name__ == "__main__":
    """
    app.config['SECRET_KEY'] = 'this_should_be_secret'
    app.run(host='127.0.0.1', port=5000, debug=True)
    """
    setting = Setting('src')
    os.makedirs(Path('src/images_all'), exist_ok=True)
    setting = Setting('src')
    prepare_data(setting)
    model, class_names = network.get_model()
    

    # # Check how model works and display results
    #
    
    image = io.imread('src/test2017/000000289594.jpg')
    plt.figure(figsize=(12, 10))
    io.imshow(image)
    plt.show()
    #
    # # Run detection
    results = model.detect([image], verbose=1)
    #
    # # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    
