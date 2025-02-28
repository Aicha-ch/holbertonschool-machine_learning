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
        try:
            self.model = K.models.load_model(model_path)
            print("loaded model !")
            except Exception as e:
                print(f"model loading error : {e}")
                exit(1)

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
        Process the model outputs to extract bounding boxes,
        confidences, and class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            sigmoid_conf = self.sigmoid(output[..., 4])
            sigmoid_prob = self.sigmoid(output[..., 5:])

            box_conf = np.expand_dims(sigmoid_conf, axis=-1)
            box_class_prob = sigmoid_prob

            b_wh = anchors * np.exp(t_wh)
            b_wh /= self.model.input.shape[1:3]

            grid = np.indices((grid_height, grid_width)).transpose(1, 2, 0)
            grid = np.expand_dims(grid, axis=2)

            b_xy = (self.sigmoid(t_xy) + grid) / [grid_width, grid_height]

            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box *= np.array([image_width, image_height, image_width, image_height])

            boxes.append(box)
            box_confidences.append(box_conf)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
