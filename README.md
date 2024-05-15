# Image classification: classify 'fields' and 'roads'
## Introduction
An image classification model based on MobileNetV2. This model is trained to classify between images of 'roads' or 'fields'
## Installation
To install required packages, run:
```bash
pip install -r requirements.txt
```
## Repo contents
### 1. classifiers
You can find the implementation of the image classifier in the 'classifiers' directory.
To initialize an instance of image classifier, run:
```python
from classifiers.image_classifier import ImageClassifier

classifier = ImageClassifier('path-to-model-directory')
```
To run model prediction, run:
```python
prediction = classifier.predict('path-to-image')
```

### 2. model
You can find the latest model in the 'model' directory. By default, trained models are saved here.

### 3. reports
You can find the image classification report, the publication report, and the publication inside this directory.

### 4. train.py
Model training script. You can run model training with this script. To show the list of all arguments, run:
```bash
python3 train.py --help
```

### 5. model_training_and_test.ipynb
The notebook that contains all of the model training process and evaluation of the model on the test dataset.
