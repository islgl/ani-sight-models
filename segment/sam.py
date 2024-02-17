import numpy as np
import onnxruntime as ort
from typing import Union, Dict
from .codec import Encoder, Decoder

class Sam:
    """Sam predict class

    This class integrate the image encoder, prompt encoder and lightweight mask decoder.

    Args:
        encoder_session: the encoder session.
        decoder_session: the decoder session.
    """

    def __init__(self,
                 encoder_session: ort.InferenceSession,
                 decoder_session: ort.InferenceSession):
        self.encoder = Encoder(encoder_session)
        self.decoder = Decoder(decoder_session)

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
                 ) -> Dict:
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
