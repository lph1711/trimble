"""Defines basics of image classifier
"""

from abc import ABC, abstractmethod
from typing import Union
from PIL import Image

class ImageClassifierModule(ABC):
    """Base module for image classifier
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def predict_image(self, image: Union[str, Image.Image]) -> str:
        """Predicts the image content

        Args:
            image(str, Image.Image): Path to image or PIL.Image instance

        Returns:
            str: image class
        """

CLASSES = [
    "fields",
    "roads"
]

class ImageClassifierException(Exception):
    """Exception from image classifier
    """
