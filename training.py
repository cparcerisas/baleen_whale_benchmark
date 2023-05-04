import datetime
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from tensorflow.keras import regularizers


import dataset

IMAGE_HEIGHT = 90
IMAGE_WIDTH = 30
N_CHANNELS = 1

CORRECTED_SAMPLES = 2500


def create_model(log_path, n_classes):
    """
    Create the model. Stores a summary of the model in the log_path, called model_summary.txt
    :param log_path: folder to store the logs
    :param n_classes: number of classes
    :return: created model
    """
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

    model_summary_path = log_path.joinpath('model_summary.txt')
    # Open the file
    with open(model_summary_path, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    return model


def train_model(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, early_stop, monitoring_metric,
                model_log_path):
    """
    Train the model. Will store the logs of the training inside the folder model_log_path in a file called logs.csv
    :param model: model created already
    :param x_train: x to train
    :param y_train: y to train
    :param x_valid: x to validate
    :param y_valid: y to validate
    :param batch_size: batch size
    :param epochs: number of epochs
    :param early_stop: number of no increase performance to stop
    :param monitoring_metric: string, metric to use to monitor performance
    :param model_log_path: folder where to store the logs
    :return: model, history
    """
    opt = tf.keras.optimizers.Adam()
    METRICS = [
        keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    ]

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=METRICS)

    model_save_filename = model_log_path.joinpath('checkpoints')

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=early_stop, restore_best_weights=True)
    mdl_checkpoint_cb = keras.callbacks.ModelCheckpoint(
        model_save_filename, save_weights_only=True, monitor=monitoring_metric, save_best_only=True
    )
    history_cb = tf.keras.callbacks.CSVLogger(model_log_path.joinpath('logs.csv'), separator=',', append=False)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid),
                        callbacks=[early_stopping_cb, mdl_checkpoint_cb, history_cb])

    return model, history


def get_model_scores(model, x_test, y_test, categories):
    """
    Test the model on x_test and y_test.

    :param model: tensorflow model
    :param x_test: X
    :param y_test: y
    :param categories: names of the categories to create the confusion matrix
    :return: scores_df (DataFrame) and con_mat_df (DataFrame)
    """
    scores = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    con_mat = confusion_matrix(y_test, y_pred)

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm, index=categories, columns=categories)
    scores_df = pd.DataFrame([scores], columns=['loss', 'accuracy'])
    return scores_df, con_mat_df


def plot_confusion_matrix(con_mat_df, save_path):
    """
    Plot the confusion matrix and save it to save_path
    :param con_mat_df: DataFramem, output from get_model_scores
    :param save_path: str or path
    :return:
    """
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
    Plot the history evolution of the metrics of the model and store the images in the folder save_path with the
    correct name indicating which fold was it run

    :param history: history, output from tensorflow
    :param save_path: folder path to store the training evolution metrics
    :param fold: which fold was it run on
    :return:
    """
    training_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
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


def create_and_train_model(log_path, n_classes, x_train, y_train, x_valid, y_valid, paths_list, config, fold):
    """
    Create and train model, from config

    :param log_path:
    :param n_classes:
    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param paths_list: df with all the paths used and which set they were used in
    :param config:
    :param fold:
    :return:
    """
    cnn_model = create_model(log_path, n_classes=n_classes)
    model_log_path = log_path.joinpath('fold%s' % fold)
    cnn_model, history = train_model(cnn_model, x_train, y_train, x_valid, y_valid, config['BATCH_SIZE'],
                                     config['EPOCHS'], config['early_stop'], config['monitoring_metric'],
                                     model_log_path=model_log_path)
    cnn_model.save(model_log_path.joinpath('model'))
    plot_training_metrics(history, save_path=log_path, fold=fold)
    return cnn_model


def test_model(cnn_model, log_path, x_test, y_test, categories, fold, noise_percentage, paths_list):
    """
    Test the model
    :param cnn_model:
    :param log_path:
    :param x_test:
    :param y_test:
    :param paths_list: df with all the paths used
    :param categories:
    :param fold:
    :return:
    """
    paths_list.to_csv(log_path.joinpath('data_used_fold%s_noise%s.csv' % (fold, noise_percentage)))
    scores_i, con_mat_df = get_model_scores(cnn_model, x_test=x_test, y_test=y_test, categories=categories)
    plot_confusion_matrix(con_mat_df, log_path.joinpath('confusion_matrix_fold%s_noise%s.png' %
                                                        (fold, noise_percentage)))

    return scores_i, con_mat_df


def create_train_and_test_model(log_path, n_classes, x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list,
                                config, categories, fold, ds):
    """
    Create the model, train it and test it according to the specifications in config
    """
    cnn_model = create_and_train_model(log_path, n_classes, x_train, y_train, x_valid, y_valid,
                                       paths_list, config, fold)

    if type(config['NOISE_RATIO_TEST']) == list:
        noise_to_test = config['NOISE_RATIO_TEST']
    else:
        noise_to_test = [config['NOISE_RATIO_TEST']]

    scores_i = pd.DataFrame()
    con_mat_df = pd.DataFrame()
    for noise in noise_to_test:
        x_test, y_test, paths_list = load_more_noise(x_test, y_test, paths_list, noise, config, ds)
        scores_noise, con_mat_noise = test_model(cnn_model, log_path, x_test, y_test, categories, fold, noise,
                                                 paths_list)
        scores_noise['noise_percentage'] = noise
        con_mat_noise['noise_percentage'] = noise
        scores_i = pd.concat([scores_i, scores_noise])
        con_mat_df = pd.concat([con_mat_df, con_mat_noise])

    return scores_i, con_mat_df


def load_more_noise(x_test, y_test, paths_list, noise, config, ds):
    if noise > config['NOISE_RATIO']:
        x_test, y_test, paths_list = ds.load_more_noise(x_test, y_test, paths_list, noise)
    else:
        print('Noise percentage lower than in training. Not considering it and testing on the trained ratio')

    return x_test, y_test, paths_list


def run_from_config(config, log_path=None):
    """
    Run a train, test set according to config. The output will be saved on the log_path folder.
    To see the structure of the output folder, check the README.

    :param config:
    :param log_path:
    :return:
    """
    tf.random.set_seed(42)

    if config['USE_CORRECTED_DATASET'] and config['SAMPLES_PER_CLASS'] > CORRECTED_SAMPLES:
        raise Exception('The SAMPLES_PER_CLASS parameter (%s) is greater than the corrected samples (%s). '
                        'Please adjust.' % (config['SAMPLES_PER_CLASS'], CORRECTED_SAMPLES))

    # Create a log directory to store all the results and parameters
    now_time = datetime.datetime.now()
    if log_path is None:
        log_path = pathlib.Path(config['OUTPUT_DIR']).joinpath(now_time.strftime('%y%m%d_%H%M%S'))
    if not log_path.exists():
        os.mkdir(str(log_path))

    json.dump(config, open(log_path.joinpath('config.json'), mode='a'))

    # Load the dataset
    ds = dataset.SpectrogramDataSet(data_dir=config['DATA_DIR'], image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                                    categories=config['CATEGORIES'], join_cat=config["CATEGORIES_TO_JOIN"],
                                    locations=config['LOCATIONS'], n_channels=N_CHANNELS,
                                    corrected=config['USE_CORRECTED_DATASET'],
                                    samples_per_class=config['SAMPLES_PER_CLASS'])
    scores = pd.DataFrame()
    con_matrix = pd.DataFrame()
    if type(config['TEST_SPLIT']) == float:
        print('Performing single train/validation/test split (random). Ony one result will be given')
        x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list = ds.load_all_dataset(
            test_size=config['TEST_SPLIT'],
            valid_size=config['VALID_SPLIT'],

            noise_ratio=config[
                'NOISE_RATIO'])
        # Create and train the model
        scores_i, con_matrix_i = create_train_and_test_model(log_path, ds.n_classes, x_train, y_train, x_valid, y_valid,
                                                             x_test, y_test, paths_list, config,
                                                             categories=ds.int2class, fold=0, ds=ds)

        # scores.loc[0] = ['random', scores_i[0], scores_i[1]]
        scores = scores_i
        con_matrix = con_matrix_i.reset_index(drop=False, names='label')

    elif type(config['TEST_SPLIT']) == int:
        print('Performing K-fold stratified cross validation with K=%s. '
              'The cross validation is done to split TRAINING/VALIDATION vs TEST, '
              'the train-validation split is done randomly. This is for better error estimation. '
              'Results are given per fold' % config['TEST_SPLIT'])
        for fold, x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list in ds.folds(
                    noise_ratio=config['NOISE_RATIO'], n_folds=config['TEST_SPLIT'], valid_size=config['VALID_SPLIT']):
            # Create and train the model
            scores_i, con_matrix_i = create_train_and_test_model(log_path, ds.n_classes, x_train, y_train, x_valid,
                                                                 y_valid, x_test, y_test, paths_list, config,
                                                                 categories=ds.int2class, fold=fold, ds=ds)

            scores_i['fold'] = fold
            scores = pd.concat([scores, scores_i], ignore_index=True)
            con_matrix_i = con_matrix_i.reset_index(drop=False, names='label')
            con_matrix_i['fold'] = fold
            con_matrix = pd.concat([con_matrix, con_matrix_i], ignore_index=True)

    else:
        print('Performing blocked cross validation for each location (leave location out). '
              'Results are given per excluded location')
        for loc in config['LOCATIONS']:
            x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list = ds.load_blocked_dataset(valid_size=config[
                                                                                             'VALID_SPLIT'],
                                                                                         noise_ratio=config[
                                                                                             'NOISE_RATIO'],
                                                                                         blocked_location=loc)
            # Create and train the model
            scores_i, con_matrix_i = create_train_and_test_model(log_path, ds.n_classes, x_train, y_train, x_valid,
                                                                 y_valid, x_test, y_test, paths_list, config,
                                                                 categories=ds.int2class, fold=loc, ds=ds)
            scores_i['fold'] = loc
            scores = pd.concat([scores, scores_i], ignore_index=True)
            con_matrix_i = con_matrix_i.reset_index(drop=False, names='label')
            con_matrix_i['fold'] = loc
            con_matrix = pd.concat([con_matrix, con_matrix_i], ignore_index=True)

    con_matrix.to_csv(log_path.joinpath('total_confusion_matrix.csv'))
    scores.to_csv(log_path.joinpath('total_scores.csv'))
    con_matrix = con_matrix.drop(columns=['fold'])
    con_matrix_avg = con_matrix.groupby('label').mean()
    plot_confusion_matrix(con_matrix_avg, save_path=log_path.joinpath('mean_confusion_matrix.png'))

    return scores, con_matrix


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define the parameters for the study
    config_file = input('Config file path:')
    if config_file == '':
        config_file = './config.json'
    # Read the config file
    f = open(config_file)
    config_json = json.load(f)
    run_from_config(config_json)
