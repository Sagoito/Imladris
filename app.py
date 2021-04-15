import sys
import os

from skimage import io

from src.data_preprocesing import prepare_data
from src.mrcnn import visualize
from src.settings import Setting
from pathlib import Path

import matplotlib.pyplot as plt
import src.coco as network

from flask import Flask
from flask import render_template, request, flash, redirect, session, url_for
from werkzeug.utils import secure_filename

from multiprocessing import Process, Queue


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


@app.route('/show_image', methods=['GET', 'POST'])
def show_image():
    filename = session['filename']

    model_process = Process(target=run_model, args=(filename,
                                                    setting,
                                                    clipboard, ))                  
    try:
        model_process.start()
        model_process.join()
    except Exception as err:
        print(err)

    filename_segmented = f"user_image/{clipboard.get()}"
    return render_template('show_image.html', name=filename_segmented)


def run_model(filename, setting, clipboard):
    file_path = os.path.join(setting.user_image, filename)
    model_class_names_tuple = network.get_model()
    model = model_class_names_tuple[0]
    class_names = model_class_names_tuple[1]

    image = io.imread(file_path)
    results = model.detect([image], verbose=1)

    r = results[0]
    image = visualize.display_instances(image, r['rois'], r['masks'],
                                        r['class_ids'], class_names,
                                        r['scores'], ret_image=True,
                                        show=False)

    filename = filename.split('.')[0]
    f_segmented = f"{filename}_segmented.jpg"
    save_path = os.path.join(setting.user_image, f_segmented)
    io.imsave(save_path, image)
    clipboard.put(f_segmented)


if __name__ == "__main__":
    global setting
    global clipboard

    """use queue for communication between processes"""
    clipboard = Queue()

    setting = Setting('src')
    os.makedirs(Path('src/images_all'), exist_ok=True)
    setting = Setting('src')

    if len(sys.argv) > 1 and sys.argv[1] == "prep":
        prepare_data(setting)

    app.config['SECRET_KEY'] = 'this_should_be_secret'
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
