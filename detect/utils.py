import cv2
import numpy as np
from typing import List, Tuple
from const import CLASSES


def bbox_restore(pred: List, image: np.ndarray):
    h, w = (640, 640)
    ori_h, ori_w, _ = image.shape
    pred[..., [0, 2]] /= w
    pred[..., [1, 3]] /= h
    pred[..., [0, 2]] *= ori_w
    pred[..., [1, 3]] *= ori_h
    return pred


def draw_img(img: np.ndarray, pred, bbox_color: Tuple = (0, 255, 0), font_color: Tuple = (255, 0, 0)):
    for box in pred:
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 2)
        cv2.putText(img, f'{cls} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color,
                    2)
    return img


def id2class(pred: List):
    for i in range(len(pred)):
        pred[i][5] = CLASSES[int(pred[i][5])]
    return pred


def non_maximum_supression(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def filter_box(org_box, conf_thres: float = 0.5, iou_thres: float = 0.5) -> List:
    org_box = np.squeeze(org_box)
    conf = org_box[..., 4] > conf_thres
    # Don't use "conf is True" here, it will cause an empty output
    box = org_box[conf == True]

    cls_cinf = box[..., 5:]
    cls = []
    for i in range(len(cls_cinf)):
        cls.append(int(np.argmax(cls_cinf[i])))
    all_cls = list(set(cls))

    output = []
    for i in range(len(all_cls)):
        curr_cls = all_cls[i]
        curr_cls_box = []

        for j in range(len(cls)):
            if cls[j] == curr_cls:
                box[j][5] = curr_cls
                curr_cls_box.append(box[j][:6])

        curr_cls_box = np.array(curr_cls_box)
        curr_cls_box = xywh2xyxy(curr_cls_box)
        curr_out_box = non_maximum_supression(curr_cls_box, iou_thres)

        for k in curr_out_box:
            output.append(curr_cls_box[k])
    output = np.array(output)
    # output = output.tolist()
    return output
