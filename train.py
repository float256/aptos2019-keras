import json
import os
from glob import glob
from zipfile import ZipFile
import pandas as pd
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from dataset_processing import download_dataset, process_csv
from model import create_model

if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    if not glob(config['dataset_folder'] + "*"):
        download_dataset(username=config["username"], token=config["token"],
                         dataset_folder=config["dataset_folder"],
                         config_folder=config["config_folder"])

        # Extracting the training dataset
        os.makedirs(config["dataset_folder"] + "train_images/")
        ZipFile(config["dataset_folder"] + "train_images.zip").extractall(
            config["dataset_folder"] + "train_images/")
        os.remove(config["dataset_folder"] + "train_images.zip")

        # Extracting test dataset
        os.makedirs(config["dataset_folder"] + "test_images/")
        ZipFile(config["dataset_folder"] + "test_images.zip").extractall(
            config["dataset_folder"] + "test_images/")
        os.remove(config["dataset_folder"] + "test_images.zip")
    train_csv = pd.read_csv(config["dataset_folder"] + "train.csv")
    train_csv = process_csv(
        dataframe=train_csv, image_column_name=config["image_column_name"],
        label_column_name=config["label_column_name"],
        folder_with_images=config["dataset_folder"] + "train_images/")

    # Creating ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1./255,
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
                                 validation_split=0.15)
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_csv, x_col=config["image_column_name"],
        y_col=config["label_column_name"], subset="training",
        batch_size=config["batch_size"], target_size=(224, 224))
    val_generator = datagen.flow_from_dataframe(
        dataframe=train_csv, x_col=config["image_column_name"],
        y_col=config["label_column_name"], subset="validation",
        batch_size=config["batch_size"], target_size=(224, 224))

    # Creating checkpoint callbacks

    if config["checkpoint_folder"]:
        try:
            os.makedirs(config["checkpoint_folder"])
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
    else:
        callbacks = []

    # Training

    model = create_model()
    model.compile(optimizer=Adam(config["learning_rate"]),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_generator),
                        validation_data=val_generator,
                        validation_steps=len(val_generator),
                        epochs=config["num_epochs"],
                        callbacks=callbacks)
