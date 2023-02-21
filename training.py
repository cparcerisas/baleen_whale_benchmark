import datetime
import json
import os
import os
import pathlib
import pickle
import shutil
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import tensorboard
import tensorflow as tf
import tensorflow.keras
from scipy import stats
from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.utils import compute_class_weight, compute_sample_weight
from sklearn.utils import shuffle
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

import dataset

IMAGE_HEIGHT = 90
IMAGE_WIDTH = 30
N_CHANNELS = 1


def create_model(logpath, n_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same",
                                     kernel_initializer='he_normal', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    model_summary_path = logpath.joinpath('model_summary.txt')
    # Open the file
    with open(model_summary_path, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    return model


def train_model(model, x_train, y_train, x_valid, y_valid, batch_size, epochs):
    opt = tf.keras.optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=["acc"])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid))

    return model, history


def test_model(model, x_test, y_test, categories, save_path):
    scores = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    con_mat = confusion_matrix(y_test, y_pred)

    f1scores = f1_score(y_test, y_pred, average=None)

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm, index=categories, columns=categories)

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path.joinpath('confusion_matrix.png'))
    plt.show()

    return scores


def plot_training_metrics(history, save_path):
    """
    Plot the history evolution of the metrics of the model
    :param history:
    :param save_path:
    :return:
    """
    training_acc = history.history['acc']
    val_acc = history.history['val_acc']
    training_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epoch_count = range(1, len(training_acc) + 1)
    plt.figure(figsize=(8, 8))
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, val_acc, 'b-')
    plt.legend(['Training acc', 'Validation acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xticks(np.arange(0, max(epoch_count), 2))
    # plt.text(10, 0.01, "Test acc: " + str(scores[1]))
    plt.grid()
    plt.savefig(save_path.joinpath('fig1.png'))
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(epoch_count, training_loss, 'g--')
    plt.plot(epoch_count, val_loss, 'r-')
    plt.legend(['Training loss', 'Validation loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yticks(np.arange(0, max(val_loss), 0.2))
    plt.xticks(np.arange(0, max(epoch_count), 2))
    plt.grid()
    plt.savefig(save_path.joinpath('fig2.png'))
    plt.show()

    pass


def plot_test_predictions(test_ds, model):
    """
    Plot the predictions made by the model in the test dataset, in folders depending if they are false positives,
    negatives or true positives or negatives
    :param test_ds:
    :param model:
    :return:
    """
    pass


def run_from_config(config, logpath=None):
    tf.random.set_seed(42)

    # Create a log directory to store all the results and parameters
    now_time = datetime.datetime.now()
    if logpath is None:
        logpath = pathlib.Path(config['OUTPUT_DIR']).joinpath(now_time.strftime('%y%m%d_%H%M%S'))
    if not logpath.exists():
        os.mkdir(str(logpath))
    json.dump(config, open(logpath.joinpath('config.json'), mode='a'))

    # Load the dataset
    ds = dataset.SpectrogramDataSet(data_dir=config['DATA_DIR'], image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                    categories=config['CATEGORIES'], locations=config['LOCATIONS'],
                                    n_channels=N_CHANNELS)
    if type(config['TEST_SPLIT']) == float:
        x_train, y_train, x_valid, y_valid, x_test, y_test = ds.load_all_dataset(test_size=config['TEST_SPLIT'],
                                                                                 valid_size=config['VALID_SPLIT'],
                                                                                 samples_per_class=config[
                                                                                     'SAMPLES_PER_CLASS'],
                                                                                 noise_ratio=config['NOISE_RATIO'])
        # Create and train the model
        cnn_model = create_model(logpath, n_classes=ds.n_classes)
        cnn_model, history = train_model(cnn_model, x_train, y_train, x_valid, y_valid, config['BATCH_SIZE'],
                                         config['EPOCHS'])
    else:
        for loc in config['LOCATIONS']:
            x_train, y_train, x_valid, y_valid, x_test, y_test = ds.load_blocked_dataset(valid_size=config[
                'VALID_SPLIT'],
                                                                                         samples_per_class=config[
                                                                                             'SAMPLES_PER_CLASS'],
                                                                                         noise_ratio=config[
                                                                                             'NOISE_RATIO'],
                                                                                         blocked_location=loc)
            # Create and train the model
            cnn_model = create_model(logpath, n_classes=ds.n_classes)
            cnn_model, history = train_model(cnn_model, x_train, y_train, x_valid, y_valid, config['BATCH_SIZE'],
                                             config['EPOCHS'])

    plot_training_metrics(history, save_path=logpath)
    scores = test_model(cnn_model, x_test=x_test, y_test=y_test, categories=config['CATEGORIES'], save_path=logpath)
    return scores


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define the parameters for the study
    config_file = './config.json'
    # Read the config file
    f = open(config_file)
    config = json.load(f)
    run_from_config(config)
