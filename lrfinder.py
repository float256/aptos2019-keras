from datetime import datetime
import json
import pickle
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from train import train


class LRFinder(Callback):

    def __init__(self, start_learning_rate, multiplier):
        self.multiplier = multiplier
        self.start_learning_rate = start_learning_rate
        self.curr_train_step_number = 1
        self.learning_rate = start_learning_rate
        self.all_learning_rates = []
        self.all_loss_values = []
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        self.learning_rate *= self.multiplier
        K.set_value(self.model.optimizer.lr, self.learning_rate)
        self.curr_train_step_number += 1
        self.all_learning_rates.append(self.learning_rate)
        self.all_loss_values.append(logs.get('loss'))


if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)
    config["num_epochs"] = 1
    config["learning_rate"] = 1e-10
    lrfinder_callback = LRFinder(1e-10, config["lrfinder_multiplier"])
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0)
    train(config, [lrfinder_callback], datagen)
    with open(f'{config["lrfinder_logs_folder"]}'
              f'{datetime.now().strftime("%Y-%m-%d--%H:%M:%S")}.pickle',
              'wb') as f:
        pickle.dump([lrfinder_callback.all_loss_values,
                     lrfinder_callback.all_learning_rates], f)
