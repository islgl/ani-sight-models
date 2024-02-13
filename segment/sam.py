import numpy as np
from typing import Union,Dict
from .codec import Encoder, Decoder
from const import ROOT
from utils import singleton

@singleton
class Sam:
    """Sam predict class

    This class integrate the image encoder, prompt encoder and lightweight mask decoder.

    Args:
        vit_model_path: the path of vit encoder.
        decoder_model_path: the prompt encoder and lightweight mask decoder path.
        device: Inference device, user can choose 'cuda' or 'cpu'. default to 'cuda'.
        warmup_epoch (int): Warmup, if set 0,the model won`t use random inputs to warmup. default to 5.
    """

    def __init__(self,
                 encoder_model_path: str = ROOT + '/models/sam_vit_b_encoder.onnx',
                 decoder_model_path: str = ROOT + '/models/sam_vit_b_decoder.onnx',
                 device: str = "cpu",
                 **kwargs):
        self.encoder = Encoder(encoder_model_path, device, **kwargs)
        self.decoder = Decoder(decoder_model_path, device, **kwargs)

        self.features = None
        self.origin_image_size = None

    def register_image(self, img: np.ndarray) -> None:
        """register input image

        This function register input image and use vit tu extract feature.

        Args:
            img (np.ndarray): the input image. The input image format must be BGR.
        """
        self.origin_image_size = img.shape
        self.features = self.encoder.run(img)

    def get_mask(self,
                 point_coords: Union[list, np.ndarray] = None,
                 point_labels: Union[list, np.ndarray] = None,
                 boxes: Union[list, np.ndarray] = None,
                 mask_input: Union[list, np.ndarray] = None
                 ) ->Dict :
        """get the segment mask

        This function input prompts to segment input image.

        Args:
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
        result = self.decoder.run(self.features,
                                  self.origin_image_size[:2],
                                  point_coords,
                                  point_labels,
                                  boxes,
                                  mask_input)
        return result
