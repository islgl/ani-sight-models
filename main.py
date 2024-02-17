import cv2
import uvicorn
import numpy as np
import onnxruntime as ort

from detect import yolo_inference
from segment import Sam, gen_prompt
from utils import bytes2img
from fastapi import FastAPI, Request
from typing import Tuple
from const import ROOT

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


# @app.get("/intialize")
def intialize():
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


@app.post("/invoke")
async def invoke(request: Request):
    request_id = request.headers.get("x-fc-request-id", "")
    print("FC Invoke Start RequestId: " + request_id)

    image_bytes = await request.body()
    if len(image_bytes) == 0:
        return {
            "status": "error",
            "message": "No image found in the request body"
        }

    try:
        image = bytes2img(image_bytes)
        mask, bboxes = inference(image)
        data = {
            "mask": mask,
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
            "message": "Inference failed"
        }

    print("FC Invoke End RequestId: " + request_id)
    return result


def inference(image: np.ndarray) -> Tuple:
    """ Inference function for the model

    Args:
        image: The input image

    Returns:
        mask: The mask of the objects in the image
        pred: The bounding boxes of the objects in the image
    """

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
    mask = masks[0][0].tolist()

    return mask, bboxes


if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)

    img_path = '/Users/lgl/code/python/code/sheep.jpg'
    intialize()
    image = cv2.imread(img_path)
    mask, bboxes = inference(image)
