import json
import os
from glob import glob
import pandas as pd
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from dataset_processing import download_dataset, process_csv
from model import create_model


def train(train_config: dict, train_callbacks: list,
          train_datagen: ImageDataGenerator):
    """This function train neural network model

    Args:
        train_config: Dictionary containing information needed for training
            Dictionary structure:
            {
                "dataset_folder": Folder with dataset. If dataset is not in the
                    folder, it is downloaded using the Kaggle API,
                "username": Username in Kaggle,
                "token": Kaggle API token,
                "config_folder": Folder for configuration for Kaggle.
                    The default "config_folder" is the home folder.
                    If the function does not download the dataset to the home
                    folder (it is denoted by "~ /"), then specify the
                    absolute path to the home folder (for example,
                    "/home/user")
                "image_column_name": Name of column with image file names in
                    train.csv file,
                "label_column_name": Name of column with labels for images in
                    train.csv file,
                "batch_size": Size of each mini-batch,
                "learning_rate": Learning rate. Number in range from 0 to 1,
                "num_epochs": Number of epochs,
            }
            The variable "config" may have other fields, they simply
                will not be taken into account
        train_callbacks: list with Keras callbacks
        train_datagen: object of type ImageDataGenerator
    """
    if not glob(train_config['dataset_folder'] + "*"):
        download_dataset(username=train_config["username"],
                         token=train_config["token"],
                         dataset_folder=train_config["dataset_folder"],
                         config_folder=train_config["config_folder"])
    train_csv = pd.read_csv(train_config["dataset_folder"] + "train.csv")
    train_csv = process_csv(
        dataframe=train_csv,
        image_column_name=train_config["image_column_name"],
        label_column_name=train_config["label_column_name"],
        folder_with_images=train_config["dataset_folder"] + "train_images/")

    # Creating ImageDataGenerator

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_csv, x_col=train_config["image_column_name"],
        y_col=train_config["label_column_name"], subset="training",
        batch_size=train_config["batch_size"], target_size=(224, 224))
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_csv, x_col=train_config["image_column_name"],
        y_col=train_config["label_column_name"], subset="validation",
        batch_size=train_config["batch_size"], target_size=(224, 224))

    # Training

    model = create_model()
    model.compile(optimizer=Adam(train_config["learning_rate"]),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    if len(val_generator) != 0:
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=len(train_generator),
                            validation_data=val_generator,
                            validation_steps=len(val_generator),
                            epochs=train_config["num_epochs"],
                            callbacks=train_callbacks)
    else:
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=len(train_generator),
                            epochs=train_config["num_epochs"],
                            callbacks=train_callbacks)


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    callbacks = []
    if config["checkpoint_folder"]:
        try:
            os.makedirs(config["checkpoint_folder"])
        except FileExistsError:
            pass
        except FileNotFoundError:
            pass
        callbacks = [
            ModelCheckpoint(
                config["checkpoint_folder"] + "best_weights--{val_loss:.6f}.h5",
                verbose=1, save_best_only=True),
            ModelCheckpoint(
                config["checkpoint_folder"] + "epoch{epoch:03d}--{"
                                              "val_loss:.6f}.h5",
                verbose=1, period=1)
        ]
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 rotation_range=15,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.01,
                                 zoom_range=[0.9, 1.25],
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='reflect',
                                 data_format='channels_last',
                                 brightness_range=[0.5, 1.5],
                                 validation_split=config["val_size"])
    train(config, callbacks, datagen)
