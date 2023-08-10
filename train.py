"""Image classification training script
"""
import os
import argparse
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import *

logging.basicConfig(level = logging.INFO)

def create_model(base_model):
    """
    Create model based on base_model with new classification layers on top
    Args:
        base_model: base pre-trained model
    Returns:
        model: New model with classification layers on top
    """
    data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2)])
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1)
    inputs = tf.keras.Input(shape=(160, 160, 3))
    layer = data_augmentation(inputs)
    layer = preprocess_input(layer)
    layer = base_model(layer, training=False)
    layer = global_average_layer(layer)
    layer = tf.keras.layers.Dropout(0.2)(layer)
    outputs = prediction_layer(layer)
    return tf.keras.Model(inputs, outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", help="Path to the dataset", required=True)
    parser.add_argument("--save-dir", help="Path to directory to save model", default="model")
    parser.add_argument("--base-lr", help="Base learning rate", default=0.0001)
    parser.add_argument("--num-epochs", help="Number of training epochs", default=10)
    parser.add_argument("--num-finetune-epochs", help = "Number of fine tune epochs", default=10)
    parser.add_argument("--batch-size", help="Batch size", default=2)
    parser.add_argument("--run-test", help="Run model evaluation on testing set", default=False)
    args = parser.parse_args()

    TRAIN_DIR = os.path.join(args.dataset_path, 'train')
    IMG_SIZE = (160,160)
    SEED = 1

    #IMPORT DATASET
    logging.info("Importing dataset")
    train_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR,
                                                                validation_split=0.2,
                                                                subset="training",
                                                                seed=SEED,
                                                                batch_size=args.batch_size,
                                                                image_size=IMG_SIZE)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR,
                                                                     validation_split=0.2,
                                                                     subset="validation",
                                                                     seed=SEED,
                                                                     batch_size=args.batch_size,
                                                                     image_size=IMG_SIZE)

    logging.info("dataset imported")
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    #CREATE BASE MODEL FROM THE MOBILENETV2 MODEL, PRETRAINED ON IMAGENET
    logging.info("dataset imported")
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    base_model.trainable = False
    model = create_model(base_model)
    logging.info("Image classification model based on MobileNetV2 created")

    #TRAINING MODEL
    logging.info("start model training")
    es = EarlyStopping(patience=3, monitor='val_loss')
    metrics = ['accuracy']
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.base_lr),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=metrics)

    trained_model = model.fit(train_dataset,
                              epochs=args.num_epochs,
                              validation_data=validation_dataset,
                              callbacks=[es])
    logging.info("finished model training")

    #FINE TUNE MODEL
    logging.info("start model fine tuning")
    base_model.trainable = True
    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.base_lr/10),
                metrics=metrics)

    total_epochs =  args.num_epochs + args.num_finetune_epochs

    final_model = model.fit(train_dataset,
                            epochs=total_epochs,
                            initial_epoch=trained_model.epoch[-1],
                            validation_data=validation_dataset,
                            callbacks=[es])
    logging.info("finished model fine tuning")

    if args.run_test:
        logging.info('Run evaluation on testing set')
        TEST_DIR = os.path.join(args.dataset_path, "test_images")
        test_dataset = tf.keras.utils.image_dataset_from_directory(TEST_DIR,
                                                                batch_size=args.batch_size,
                                                                image_size=IMG_SIZE)
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
        loss, accuracy = model.evaluate(test_dataset)
        logging.info('Test accuracy : %f', accuracy)

    #SAVE MODEL
    model.save(args.save_dir)
    logging.info("Model is saved at: %s", args.save_dir)
