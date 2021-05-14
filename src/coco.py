import os
import sys
import time

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np

# Root directory of the project
from pycocotools.cocoeval import COCOeval

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
    def load_coco(self, subset, return_coco=False):
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

        count = 0
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
                count += 1
        if return_coco:
            return coco, count

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


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


"""
    Function training model with pre-trained weights
"""


def train(command="train"):
    dataset_train = CocoDataset()
    dataset_train.load_coco("train")
    dataset_train.prepare()

    dataset_val = CocoDataset()
    dataset_val.load_coco("val")
    dataset_val.prepare()
    config = CocoConfig()

    if command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir='logs/')
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs/')

    model_path = "src/mask_rcnn_coco.h5"

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)
    if command == "train":
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
    else:
        # Validation dataset
        dataset_val = CocoDataset()
        val_type = "val"
        coco, limit = dataset_val.load_coco(val_type, return_coco=True)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(limit))


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
