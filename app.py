from skimage import io

from src.data_preprocesing import prepare_data
from src.mrcnn import visualize
from src.settings import Setting
import os
from pathlib import Path

import matplotlib.pyplot as plt
import src.coco as network

if __name__ == "__main__":
    os.makedirs(Path('src/images_all'), exist_ok=True)
    setting = Setting('src')
    prepare_data(setting)
    model, class_names = network.get_model()


    # # Check how model works and display results
    #
    # image = io.imread('src/test2017/000000289594.jpg')
    # plt.figure(figsize=(12, 10))
    # io.imshow(image)
    # plt.show()
    #
    # # Run detection
    # results = model.detect([image], verbose=1)
    #
    # # Visualize results
    # r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])

