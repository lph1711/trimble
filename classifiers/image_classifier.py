"""Image Classifier predictor
"""
import logging
from typing import Union
from copy import deepcopy
from PIL import Image
import numpy as np
import tensorflow as tf
from classifiers.image_classifier_base import (ImageClassifierModule,
                                               ImageClassifierException,
                                               CLASSES)
class ImageClassifierPredictor:
    """Image classification Predictor instance
    """

    def __init__(self, model_dir) -> None:
        self._model = tf.keras.models.load_model(model_dir)

    def predict(self, image):
        """Predicts the image content

        Args:
            image(list): image

        Returns:
            str: image class

        """
        prediction = self._model.predict(np.array(image))
        prediction = tf.nn.sigmoid(prediction)
        prediction = tf.where(prediction < 0.5, 0, 1)
        return CLASSES[prediction.numpy()[0][0]]

class ImageClassifier(ImageClassifierModule):
    """Image classification model
    """
    def __init__(self, model_dir) -> None:
        self._predictor = ImageClassifierPredictor(model_dir)
        self._logger = logging.getLogger("ImageClassifier")
        self._logger.info("Initializing model from %s", model_dir)

    def predict_image(self, image: Union[str, Image.Image]) -> str:
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as exc:
                raise ImageClassifierException("Image path not found") from exc
        image = image.resize((160, 160))
        img = np.asarray(image)
        img_2 = deepcopy(img)
        images = [img, img_2]
        return self._predictor.predict(images)
