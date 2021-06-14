import sys
import os
import threading
import time

import cv2
import json
import numpy as np
from random import randint
from skimage import io

from src.data_preprocesing import prepare_data
from src.mrcnn import visualize
from src.settings import Setting
from pathlib import Path
from generative_inpainting.test import run_gan

import src.coco as network

from flask import Flask
from flask import render_template, request, flash, redirect, session, url_for
from werkzeug.utils import secure_filename

from multiprocessing import Process, Queue, Manager, Value

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


@app.before_first_request
def before_first_request():

    model_process = Process(target=run_model, args=(setting,
                                                    clipboard,
                                                    run_statement))
    try:
        model_process.start()
        # model_process.join()
    except Exception as err:
        print(err)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_correct_filename(filename: str) -> bool:
    counter = 0
    for ch in filename:
        if ch == '.':
            counter += 1

    if counter == 0 or counter > 1:
        return False
    else:
        list_of_two = filename.split(sep=".")
        if list_of_two[0] == '':
            return False
        return True


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

        if not is_correct_filename(file.filename):
            flash('Bad file name')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            """
            Add random number to filename for avoiding collisions.
            """
            filename = str(randint(0, 1000000)) + secure_filename(file.filename)
            file.save(os.path.join(setting.user_image, filename))
            session['filename'] = filename

            split = session['filename'].split('.')
            session['filename_segmented'] = f"{split[0]}_segmented.{split[1]}"
            session['filename_masked'] = f"masked_{session['filename']}"
            session['filename_out'] = f"out_{session['filename']}"

            names = [
                session['filename'],
                session['filename_segmented'],
                session['filename_masked'],
                session['filename_out'],
            ]
            """
            If user quit application before last request
            his images will be removed after five minutes 
            """
            delete = threading.Thread(target=remove_images, args=(names, setting, 60 * 5))
            delete.start()

            return redirect(url_for('show_image'))
        else:
            flash("Bad extension allowed extensions jpg, png and jpeg")
            return redirect(request.url)

    return render_template('index.html')


def remove_images(names: list, setting, sec: int) -> None:
    """
    Args:
        names:
        setting:
        sec: defining time for deleting user images
    Description:
        Removing redundant objects from server after ten seconds
    """
    time.sleep(sec)
    try:
        os.remove(os.path.join(setting.user_image, names[0]))
        os.remove(os.path.join(setting.user_image, names[1]))
        os.remove(os.path.join(setting.user_image, names[2]))
        os.remove(os.path.join(setting.user_image, names[3]))
    except Exception as e:
        print(e)


@app.route('/show_inpainted_image')
def show_inpainted_image():
    filename_out = clipboard.get()
    names = [
        session['filename'],
        session['filename_segmented'],
        session['filename_masked'],
        session['filename_out'],
    ]
    """
    After ten seconds from user images will be removed
    """
    delete = threading.Thread(target=remove_images, args=(names, setting, 10))
    delete.start()
    return render_template('show_image.html', name=filename_out)


def prepare_masked_image():
    filename = session['filename']
    det = clipboard.get()
    cords = request.form['cords']
    cords = json.loads(cords)
    go = False
    chosen_object_pixels = None
    for key, val in det.items():
        if cords['y_cord'] in val:
            for x_cord in val[cords['y_cord']]:
                if x_cord == cords['x_cord']:
                    chosen_object_pixels = val #if chosen coordinates belongs to any object, we get back a dictionary with its coordinates
                    go = True
                    break
        if go:
            break
    try:
        img = cv2.imread("./static/user_image/"+filename, cv2.IMREAD_COLOR)
        masked_file = np.zeros((img.shape[0], img.shape[1], 4))

        for row in chosen_object_pixels:
            for column in chosen_object_pixels[row]:
                masked_file[row-15:row+15, column-15:column+15, :] = 255

        cv2.imwrite("./static/user_image/masked_"+filename, masked_file)
        session['filename_masked'] = "masked_" + filename
        session['filename_out'] = "out_" + filename
        run_gan("./generative_inpainting/model_logs/release_places2_256",
                "./static/user_image/" + filename,
                "./static/user_image/masked_" + filename,
                "./static/user_image/out_" + filename)
    except Exception as e:
        print(e)


@app.route('/show_image', methods=['GET', 'POST'])
def show_image():

    if request.method == 'POST':
        prepare_masked_image()
        filename_out = "user_image/out_" + session['filename']
        clipboard.put(filename_out)
        return redirect(url_for('show_inpainted_image'))

    clipboard.put(session['filename'])
    run_statement.value = True

    while True:
        if not run_statement.value:
            break
    filename_segmented = f"{clipboard.get()}"
    session['filename_segmented'] = filename_segmented
    filename_segmented = "user_image/" + filename_segmented
    """           
    dictionary like {(detected object category from r['class_ids']): {y coordinates : [x coordinates in y_cords line]}}
    example : 1 - number if 'person' object detected and choosen
    {1: {720:[1078, 1079, 1080]}} -> cords (720,1078),(720,1079),(720,1080) belong to area where detected object 1
    it's the same dictionary like "detected_objects" from "run_model" thread
    Now "choosen_object_pixels" is a dict of pixels in choosen object's area or None if (cords['y_cords'],cords['x_cords'])
    are not from any object area
    And now we can use that to remove this obbject form picture     

    << DEVELOP STUFF TO HELP IN DEBUG BUT TO REMOVE IN FINAL VERSION >>        
    if choosen_object_pixels != None: # if choosen coords are in any object area                       
        print(choosen_object_pixels)
    else:
        print("XY cords are form area not bounded with any detected object")        
    """
    return render_template('show_image.html', name=filename_segmented)


def run_model(setting, clipboard, run_statement):
    model_class_names_tuple = network.get_model()
    model = model_class_names_tuple[0]
    class_names = model_class_names_tuple[1]
    print("Weights loaded")
    while True:
        if run_statement.value:
            filename = clipboard.get()
            file_path = os.path.join(setting.user_image, filename)

            image = io.imread(file_path)
            if len(image.shape) > 2 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            results = model.detect([image], verbose=1)

            r = results[0]
            image = visualize.display_instances(image, r['rois'], r['masks'],
                                                r['class_ids'], class_names,
                                                r['scores'], ret_image=True,
                                                show=False)
            x_px = 0
            y_px = 0
            i_category = 0
            detected_objects = {}
            for obj in r['class_ids']:
                detected_objects[i_category] = {}
                i_category +=1
            for y in r['masks']:
                x_px = 0
                for x in y:
                    i_category = 0
                    for if_detected in x:
                        if if_detected == True:
                            if y_px not in detected_objects[i_category]:
                                detected_objects[i_category][y_px]=[]
                            detected_objects[i_category][y_px].append(x_px)
                        i_category+=1
                    x_px+=1
                y_px+=1

            filename = filename.split('.')[0]
            f_segmented = f"{filename}_segmented.jpg"
            save_path = os.path.join(setting.user_image, f_segmented)
            io.imsave(save_path, image)
            clipboard.put(f_segmented)
            clipboard.put(detected_objects)
            run_statement.value = False


if __name__ == "__main__":
    global setting
    global clipboard
    run_statement = Value("i", False)

    """use queue for communication between processes"""
    clipboard = Manager().Queue()
    setting = Setting('src')
    os.makedirs(Path('src/images_all'), exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "prep":
        prepare_data(setting)

    app.config['SECRET_KEY'] = 'this_should_be_secret'
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
