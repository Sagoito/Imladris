import os
import sys
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from src.mrcnn import model as modellib, utils
from src.mrcnn.config import Config


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = 'coco'
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class CocoDataset(utils.Dataset):
    def load_coco(self, subset):
        coco = COCO("src/annotations/instances_val2017.json")
        image_dir = "src/{}2017".format(subset)

        # Load classes
        class_ids = sorted(coco.getCatIds())

        # All images
        image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        image_list = os.listdir(image_dir)

        for i in image_ids:
            path = coco.imgs[i]['file_name']
            if image_list.count(path):
                self.add_image(
                    "coco", image_id=i,
                    path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    annotations=coco.loadAnns(coco.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

"""
    Function training model with pre-trained weights
"""
def train():
    dataset_train = CocoDataset()
    dataset_train.load_coco("train")
    dataset_train.prepare()

    dataset_val = CocoDataset()
    dataset_val.load_coco("val")
    dataset_val.prepare()
    config = CocoConfig()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir='logs/')

    model_path = "src/mask_rcnn_coco.h5"
    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all')

    return model

"""
    Function returns model and class_names from pre-trained weights
"""
def get_model():
    dataset = CocoDataset()
    dataset.load_coco("test")
    dataset.prepare()

    config = CocoConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs/')

    model_path = "src/mask_rcnn_coco.h5"
    # Load weights
    print("Loading weights ", model_path)

    model.load_weights(model_path, by_name=True)

    return model, dataset.class_names
