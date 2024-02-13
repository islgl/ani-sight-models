import json
import uvicorn
import numpy as np

from detect import yolo_inference
from segment import Sam, gen_prompt
from utils import bytes2img
from fastapi import FastAPI, Request, Response
from typing import Tuple

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "AniSight inference service is running."}


@app.get("/invoke")
async def invoke(request: Request):
    request_id = request.headers.get("x-fc-request-id", "")
    print("FC Invoke Start RequestId: " + request_id)

    image_bytes = await request.body()
    image = bytes2img(image_bytes)

    try:
        masks, bboxes = inference(image)
        result = {
            "masks": masks,
            "bboxes": bboxes
        }
        msg = 'success'
        status_code = 200
    except Exception:
        result = {
            "error": "An error occurred during inference"
        }
        msg = 'failed'
        status_code = 500

    print("FC Invoke End RequestId: " + request_id)
    return Response(content=json.dumps(result),
                    status_code=status_code,
                    media_type="application/json",
                    headers={"request-id": request_id, "msg": msg})


def inference(image: np.ndarray) -> Tuple:
    """ Inference function for the model

    Args:
        image: The input image

    Returns:
        masks: The masks of the objects in the image
        pred: The bounding boxes of the objects in the image
    """

    # YOLO inference
    bboxes = yolo_inference(image)

    # SAM inference
    sam = Sam(device='cuda')
    prompt = gen_prompt(bboxes)
    sam.register_image(image)
    masks = sam.get_mask(boxes=prompt)['masks']
    masks = masks > 0
    masks = masks.tolist()

    return masks, bboxes


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
