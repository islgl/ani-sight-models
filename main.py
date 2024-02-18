import cv2
import uvicorn
import os
import numpy as np
import onnxruntime as ort

from detect import yolo_inference, draw_img
from segment import Sam, gen_prompt, apply_mask
from fastapi import FastAPI
from typing import Tuple
from const import ROOT, OSS_PATH
from utils import str2tuple 

app = FastAPI()

global yolo_ort_session
global sam_enc_ort_session
global sam_dec_ort_session


@app.get("/")
async def root():
    return {
        "status": "success",
        "message": "Server is running!",
    }



def initialize():
    # Load models
    yolo_weights_path = ROOT + '/models/yolov5s-animal-sim.onnx'
    sam_enc_weights_path = ROOT + '/models/sam_vit_b_encoder.onnx'
    sam_dec_weights_path = ROOT + '/models/sam_vit_b_decoder.onnx'

    global yolo_ort_session
    global sam_enc_ort_session
    global sam_dec_ort_session

    provider = ['CUDAExecutionProvider']

    yolo_ort_session = ort.InferenceSession(yolo_weights_path, providers=provider)
    sam_enc_ort_session = ort.InferenceSession(sam_enc_weights_path, providers=provider)
    sam_dec_ort_session = ort.InferenceSession(sam_dec_weights_path, providers=provider)


@app.get("/invoke")
async def invoke(image_id: int,
                 image_name: str,
                 bbox_color: str='(0, 255, 0)',
                 font_color: str='(255, 0, 0)'):
    image_path = os.path.join(OSS_PATH, 'images', image_name)
    bbox_color = str2tuple(bbox_color)
    font_color = str2tuple(font_color)

    if not os.path.exists(image_path):
        result = {
            "status": "error",
            "message": "Image not found",
            "data": {
                "image_id": image_id,
                "image_name": image_name
            }
        }
        return result

    try:
        image = cv2.imread(image_path)
        mask, bboxes = inference(image)

        mask_name = image_name.split('.')[0] + '.png'

        mask_path = os.path.join(OSS_PATH, 'masks', mask_name)
        label_path = os.path.join(OSS_PATH, 'labels', image_name)

        print('draw masked image...')
        masked_img = apply_mask(image, mask)

        print('draw labeled image...')
        labeled_img = draw_img(image, bboxes, bbox_color, font_color)

        cv2.imwrite(mask_path, masked_img)
        print('save labeled image...')

        cv2.imwrite(label_path, labeled_img)
        print('save masked image...')

        data = {
            "image_id": image_id,
            "image_name": image_name,
            "mask_name": mask_name,
            "label_name": image_name,
            "bboxes": bboxes
        }
        result = {
            "status": "success",
            "message": "Inference successful",
            "data": data
        }

    except Exception:
        result = {
            "status": "error",
            "message": "Inference failed",
            "data": {
                "image_id": image_id,
                "image_name": image_name
            }
        }

    return result


def inference(image: np.ndarray) -> Tuple:
    """ Inference function for the model

    Args:
        image: The input image

    Returns:
        mask: The mask of the objects in the image
        pred: The bounding boxes of the objects in the image
    """

    initialize()

    global yolo_ort_session
    global sam_enc_ort_session
    global sam_dec_ort_session

    # YOLO inference
    bboxes = yolo_inference(image, yolo_ort_session)

    if not bboxes:
        return [], []

    # SAM inference
    sam = Sam(sam_enc_ort_session, sam_dec_ort_session)
    prompt = gen_prompt(bboxes)
    sam.register_image(image)
    masks = sam.get_mask(boxes=prompt)['masks']
    mask = masks[0][0]

    return mask, bboxes


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
