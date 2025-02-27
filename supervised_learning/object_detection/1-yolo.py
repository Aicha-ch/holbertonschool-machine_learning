#!/usr/bin/env python3
"""
Initialize Yolo
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo v3 algorithm for object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
    """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
    """
    Process the model outputs to extract bounding boxes
    confidences, and class probabilities.
    """
    boxes, box_confidences, box_class_probs = [], [], []

    input_size = np.array(self.model.inputs[0].shape.as_list()[1:3])
    image_size = np.flip(image_size)

    for i, output in enumerate(outputs):
        anchors = self.anchors[i]
        grid_h, grid_w = output.shape[:2]

        t_xy, t_wh = output[..., :2], output[..., 2:4]
        box_conf = np.expand_dims(self.sigmoid(output[..., 4]), axis=-1)
        box_class_prob = self.sigmoid(output[..., 5:])

        b_wh = (anchors * np.exp(t_wh)) / input_size
        grid = np.indices((grid_w, grid_h)).transpose(1, 2, 0)
        grid = np.tile(grid[..., np.newaxis, :], (1, 1, anchors.shape[0], 1))

        b_xy = (self.sigmoid(t_xy) + grid) / [grid_w, grid_h]
        b_xy1, b_xy2 = b_xy - (b_wh / 2), b_xy + (b_wh / 2)
        

        box = np.concatenate((b_xy1, b_xy2), axis=-1) * np.tile(input_size, 2)

        boxes.append(box)
        box_confidences.append(box_conf)
        box_class_probs.append(box_class_prob)

    return boxes, box_confidences, box_class_probs
