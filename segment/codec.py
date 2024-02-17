import onnxruntime as ort
import numpy as np
import cv2
from tqdm import tqdm
from typing import Union

from .utils import apply_coords, apply_boxes


class Encoder:
    """ SAM image encoder, which can extract feature from input image.

    Args:
        session: the onnxruntime session of SAM encoder.
    """

    def __init__(self,
                 session: ort.InferenceSession):
        print("loading encoder model...")
        self.session = session

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape

        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        """extract image feature

        this function can use Encoder to extract feature from transformed image.

        Args:
            tensor (np.ndarray): input image.

        Returns:
            np.ndarray: image`s feature.
        """
        assert list(tensor.shape) == self.input_shape
        feature = self.session.run(None, {self.input_name: tensor})[0]
        assert list(feature.shape) == self.output_shape
        return feature

    def transform(self, img: np.ndarray) -> np.ndarray:
        """image transform

        This function can convert the input image to the required input format for Encoder.

        Args:
            img (np.ndarray): input image, the image type should be BGR.

        Returns:
            np.ndarray: transformed image.
        """
        h, w, c = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.mean) / self.std

        size = max(h, w)
        img = np.pad(img, ((0, size - h), (0, size - w), (0, 0)), 'constant', constant_values=(0, 0))
        img = cv2.resize(img, self.input_shape[2:])
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, axes=[0, 3, 1, 2]).astype(np.float32)
        return img

    def run(self, img: np.ndarray) -> np.ndarray:
        """Encoder forward function

        This function can transform the input image and
        use Encoder to extract feature from input image.

        Returns:
            np.ndarray: the input image`s feature.
        """
        tensor = self.transform(img)
        features = self._extract_feature(tensor)
        return features


class Decoder:
    """Sam decoder model.

    This class is the sam prompt encoder and lightweight mask decoder.

    Args:
        session: the onnxruntime session of SAM decoder.
    """
    img_size = (1024, 1024)
    mask_threshold = 0.0

    def __init__(self,
                 session: ort.InferenceSession):

        print("loading decoder model...")
        self.session = session

    def run(self,
            img_embeddings: np.ndarray,
            origin_image_size: Union[list, tuple],
            point_coords: Union[list, np.ndarray] = None,
            point_labels: Union[list, np.ndarray] = None,
            boxes: Union[list, np.ndarray] = None,
            mask_input: np.ndarray = None) -> dict:
        """decoder forward function

        This function can use image feature and prompt to generate mask. Must input
        at least one box or point.

        Args:
            img_embeddings (np.ndarray): the image feature from vit encoder.
            origin_image_size (list or tuple): the input image size.
            point_coords (list or np.ndarray): the input points.
            point_labels (list or np.ndarray): the input points label, 1 indicates
                a foreground point and 0 indicates a background point.
            boxes (list or np.ndarray): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray): A low resolution mask input to the model,
                typically coming from a previous prediction iteration. Has form
                1xHxW, where for SAM, H=W=256.

        Returns:
            dict: the segment results.
        """
        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")
        if mask_input is None:
            mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask_input = np.zeros(1, dtype=np.float32)
        else:
            mask_input = np.expand_dims(mask_input, axis=0)
            has_mask_input = np.ones(1, dtype=np.float32)
            if mask_input.shape != (1, 1, 256, 256):
                raise ValueError("Got wrong mask!")
        if point_coords is not None:
            if isinstance(point_coords, list):
                point_coords = np.array(point_coords, dtype=np.float32)
            if isinstance(point_labels, list):
                point_labels = np.array(point_labels, dtype=np.float32)

        if point_coords is not None:
            point_coords = apply_coords(point_coords, origin_image_size, self.img_size[0]).astype(np.float32)
            point_coords = np.expand_dims(point_coords, axis=0)
            point_labels = np.expand_dims(point_labels, axis=0)

        if boxes is not None:
            if isinstance(boxes, list):
                boxes = np.array(boxes, dtype=np.float32)
            assert boxes.shape[-1] == 4

            boxes = apply_boxes(boxes, origin_image_size, self.img_size[0]).reshape((1, -1, 2)).astype(np.float32)
            box_label = np.array([[2, 3] for i in range(boxes.shape[1] // 2)], dtype=np.float32).reshape((1, -1))

            if point_coords is not None:
                point_coords = np.concatenate([point_coords, boxes], axis=1)
                point_labels = np.concatenate([point_labels, box_label], axis=1)
            else:
                point_coords = boxes
                point_labels = box_label

        assert point_coords.shape[0] == 1 and point_coords.shape[-1] == 2
        assert point_labels.shape[0] == 1

        input_dict = {"image_embeddings": img_embeddings,
                      "point_coords": point_coords,
                      "point_labels": point_labels,
                      "mask_input": mask_input,
                      "has_mask_input": has_mask_input,
                      "orig_im_size": np.array(origin_image_size, dtype=np.float32)}
        res = self.session.run(None, input_dict)

        result_dict = dict()
        for i in range(len(res)):
            out_name = self.session.get_outputs()[i].name
            if out_name == "masks":
                mask = (res[i] > self.mask_threshold).astype(np.int32)
                result_dict[out_name] = mask
            else:
                result_dict[out_name] = res[i]

        return result_dict
