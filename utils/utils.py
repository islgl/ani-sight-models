import base64
import cv2
import numpy as np


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def bytes2img(file: bytes) -> np.ndarray:
    """ Convert file to image

    Args:
        file: file to be converted

    Returns:
        The image converted from the file
    """

    nparr = np.frombuffer(file, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def base64enc(image: np.ndarray) -> str:
    """ Convert image to base64 string

    Args:
        image: image to be converted

    Returns:
        The base64 string of the image
    """

    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64dec(base64str: str) -> np.ndarray:
    """ Convert base64 string to image

    Args:
        base64str: The base64 string of the image

    Returns:
        The image converted from the base64 string
    """

    imgdata = base64.b64decode(base64str)
    return cv2.imdecode(np.fromstring(imgdata, np.uint8), 1)
