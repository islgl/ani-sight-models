import numpy as np
from .yolo import Yolo
from .utils import bbox_restore, id2class, filter_box
from typing import List


def yolo_inference(image: np.ndarray) -> List:
    """ YOLO inference

    Args:
        image: The image to be detected

    Returns:
        The detected results [[x1, y1, x2, y2, conf, cls], ...]
    """

    yolo = Yolo()
    pred = yolo.inference(image)
    pred = filter_box(pred)
    pred = bbox_restore(pred, image)

    pred = pred.tolist()
    for i in range(len(pred)):
        pred[i][:4] = [int(x) for x in pred[i][:4]]
        pred[i][4] = round(pred[i][4], 2)
    pred = id2class(pred)

    return pred
