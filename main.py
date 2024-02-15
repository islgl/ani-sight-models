import uvicorn
import numpy as np

from detect import yolo_inference
from segment import Sam, gen_prompt
from utils import bytes2img
from fastapi import FastAPI, Request, Response
from typing import Tuple
from response import CustomResponse

app = FastAPI()


@app.get("/")
async def root():
   return {
        "status": "success",
        "message": "Server is running!",
   }


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
        result={
            "status": "success",
            "message": "Inference successful",
            "data": data
        }
        
    except Exception:
       result={
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

    # YOLO inference
    bboxes = yolo_inference(image)

    if bboxes == []:
        return [], []

    # SAM inference
    sam = Sam(device='cuda')
    prompt = gen_prompt(bboxes)
    sam.register_image(image)
    masks = sam.get_mask(boxes=prompt)['masks']
    mask=masks[0][0].tolist()

    return mask, bboxes


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
