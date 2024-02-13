import cv2
import os
import onnxruntime as ort
import numpy as np
from const import ROOT
from utils import singleton


@singleton
class Yolo:
    """ YOLOv5s-Animal

    Attributes:
        onnx_session: The onnxruntime session
    """

    def __init__(self, weights_path: str = ROOT + '/models/yolov5s-animal-sim.onnx'):
        if not weights_path.endswith('.onnx'):
            raise ValueError('Weights file must be .onnx file')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'Weights file not found: {weights_path}')

        self.onnx_session = ort.InferenceSession(weights_path)
        self._input_name, self._output_name = self._get_input_output_names()
        self._input_size = (640, 640)

    def _get_input_output_names(self):
        input_names = [node.name for node in self.onnx_session.get_inputs()]
        output_names = [node.name for node in self.onnx_session.get_outputs()]
        return input_names, output_names

    def _get_input_feed(self, image: np.ndarray):
        input_feed = {name: image for name in self._input_name}
        return input_feed

    def inference(self, image: np.ndarray):
        """ YOLO inference

        Args:
            image: The image to be detected

        Returns:
            The detected results and the original image
        """
        try:
            ori_img = cv2.resize(image, self._input_size)
            img = ori_img[:, :, ::-1].transpose(2, 0, 1)
            img = img.astype(dtype=np.float32)
            img /= 255.0
            img = np.expand_dims(img, axis=0)
            input_feed = self._get_input_feed(img)
            pred = self.onnx_session.run(None, input_feed)[0]
            return pred
        except Exception as e:
            raise RuntimeError(f'Error during inference: {str(e)}')
