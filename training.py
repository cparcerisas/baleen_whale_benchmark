import datetime
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

import dataset

IMAGE_HEIGHT = 90
IMAGE_WIDTH = 30
N_CHANNELS = 1

CORRECTED_SAMPLES = 2500


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


def test_model(model, x_test, y_test, categories):
    scores = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    con_mat = confusion_matrix(y_test, y_pred)

    f1scores = f1_score(y_test, y_pred, average=None)

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm, index=categories, columns=categories)

    return scores, con_mat_df


def plot_confusion_matrix(con_mat_df, save_path):
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.savefig(save_path)
    plt.show()


def plot_training_metrics(history, save_path, fold):
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
    plt.savefig(save_path.joinpath('training_accuracy_fold_%s.png' % fold))
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
    plt.savefig(save_path.joinpath('training_loss_fold_%s.png' % fold))
    plt.show()

    pass


def plot_test_predictions(test_ds, model):
    """
    Plot the predictions made by the model in the test dataset, in folders depending on if they are false positives,
    negatives or true positives or negatives
    :param test_ds:
    :param model:
    :return:
    """
    pass


def create_train_and_test_model(logpath, n_classes, x_train, y_train, x_valid, y_valid, x_test, y_test,
                                config, categories, fold):
    cnn_model = create_model(logpath, n_classes=n_classes)
    cnn_model, history = train_model(cnn_model, x_train, y_train, x_valid, y_valid, config['BATCH_SIZE'],
                                     config['EPOCHS'])

    plot_training_metrics(history, save_path=logpath, fold=fold)
    scores_i, con_mat_df = test_model(cnn_model, x_test=x_test, y_test=y_test, categories=categories)
    plot_confusion_matrix(con_mat_df, logpath.joinpath('confusion_matrix_fold_%s.png' % fold))

    return scores_i, con_mat_df


def run_from_config(config, logpath=None):
    tf.random.set_seed(42)

    if config['USE_CORRECTED_DATASET'] and config['SAMPLES_PER_CLASS'] > CORRECTED_SAMPLES:
        raise Exception('The SAMPLES_PER_CLASS parameter (%s) is greater than the corrected samples (%s). '
                        'Please adjust.' % (config['SAMPLES_PER_CLASS'], CORRECTED_SAMPLES))

    # Create a log directory to store all the results and parameters
    now_time = datetime.datetime.now()
    if logpath is None:
        logpath = pathlib.Path(config['OUTPUT_DIR']).joinpath(now_time.strftime('%y%m%d_%H%M%S'))
    if not logpath.exists():
        os.mkdir(str(logpath))
    json.dump(config, open(logpath.joinpath('config.json'), mode='a'))

    # Load the dataset
    ds = dataset.SpectrogramDataSet(data_dir=config['DATA_DIR'], image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                    categories=config['CATEGORIES'], join_cat=config["CATEGORIES_TO_JOIN"],
                                    locations=config['LOCATIONS'], n_channels=N_CHANNELS,
                                    corrected=config['USE_CORRECTED_DATASET'])
    scores = pd.DataFrame(columns=['test_fold', 'loss', 'accuracy'])
    con_matrix = pd.DataFrame()
    if type(config['TEST_SPLIT']) == float:
        print('Performing single train/validation/test split (random). Ony one result will be given')
        x_train, y_train, x_valid, y_valid, x_test, y_test = ds.load_all_dataset(test_size=config['TEST_SPLIT'],
                                                                                 valid_size=config['VALID_SPLIT'],
                                                                                 samples_per_class=config[
                                                                                     'SAMPLES_PER_CLASS'],
                                                                                 noise_ratio=config['NOISE_RATIO'])
        # Create and train the model
        scores_i, con_matrix_i = create_train_and_test_model(logpath, ds.n_classes, x_train, y_train, x_valid, y_valid,
                                                             x_test, y_test, config, categories=ds.int2class, fold=0)

        scores.loc[0] = ['random', scores_i[0], scores_i[1]]
        con_matrix = con_matrix_i.reset_index(drop=False, names='label')

    elif type(config['TEST_SPLIT']) == int:
        print('Performing K-fold stratified cross validation with K=%s. The cross validation is done in the TEST set, '
              'The train-validation split is done randomly. This is for better error estimation. '
              'Results are given per fold' % config['TEST_SPLIT'])
        x, y = ds.load_data(config['SAMPLES_PER_CLASS'], config['NOISE_RATIO'], locations_to_exclude=None)
        kfold = StratifiedKFold(n_splits=config['TEST_SPLIT'], shuffle=True)
        for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
            x_model = x[train_index]
            y_model = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
            x_train, x_valid, y_train, y_valid = train_test_split(x_model, y_model,
                                                                  test_size=config['VALID_SPLIT'], shuffle=True)

            # Create and train the model
            scores_i, con_matrix_i = create_train_and_test_model(logpath, ds.n_classes, x_train, y_train, x_valid,
                                                                 y_valid, x_test, y_test, config,
                                                                 categories=ds.int2class, fold=fold)
            scores.loc[len(scores)] = [fold, scores_i[0], scores_i[1]]
            con_matrix_i = con_matrix_i.reset_index(drop=False, names='label')
            con_matrix_i['fold'] = fold
            con_matrix = pd.concat([con_matrix, con_matrix_i], ignore_index=True)

    else:
        print('Performing blocked cross validation for each location (leave location out). '
              'Results are given per excluded location')
        for loc in config['LOCATIONS']:
            x_train, y_train, x_valid, y_valid, x_test, y_test = ds.load_blocked_dataset(valid_size=config[
                                                                                             'VALID_SPLIT'],
                                                                                         samples_per_class=config[
                                                                                             'SAMPLES_PER_CLASS'],
                                                                                         noise_ratio=config[
                                                                                             'NOISE_RATIO'],
                                                                                         blocked_location=loc)
            # Create and train the model
            scores_i, con_mat_norm = create_train_and_test_model(logpath, ds.n_classes, x_train, y_train, x_valid,
                                                                 y_valid, x_test, y_test, config,
                                                                 categories=ds.int2class, fold=loc)
            scores.loc[len(scores)] = [loc, scores_i[0], scores_i[1]]
            con_matrix_i = con_matrix_i.reset_index(drop=False, names='label')
            con_matrix_i['excluded_loc'] = loc
            con_matrix = pd.concat([con_matrix, con_matrix_i], ignore_index=True)

    con_matrix.to_csv(logpath.joinpath('total_confusion_matrix.csv'))
    con_matrix = con_matrix.drop(columns=['fold'])
    con_matrix_avg = con_matrix.groupby('label').mean()
    plot_confusion_matrix(con_matrix_avg, save_path=logpath.joinpath('mean_confusion_matrix.png'))

    return scores, con_matrix


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define the parameters for the study
    config_file = './config.json'
    # Read the config file
    f = open(config_file)
    config = json.load(f)
    run_from_config(config)
