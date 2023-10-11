import pathlib
import pandas as pd
import tensorflow as tf
import json
import datetime
import os

import training
import dataset
import model


def run_from_config(config_path, log_path=None):
    """
    Run a train, test set according to config. The output will be saved on the log_path folder.
    To see the structure of the output folder, check the README.

    :param config_path:
    :param log_path:
    :return:
    """
    tf.random.set_seed(42)

    # Read the config file
    f = open(config_path)
    config = json.load(f)

    if config['USE_CORRECTED_DATASET'] and config['SAMPLES_PER_CLASS'] > dataset.CORRECTED_SAMPLES:
        raise Exception('The SAMPLES_PER_CLASS parameter (%s) is greater than the corrected samples (%s). '
                        'Please adjust.' % (config['SAMPLES_PER_CLASS'], dataset.CORRECTED_SAMPLES))

    # Create a log directory to store all the results and parameters
    now_time = datetime.datetime.now()
    if log_path is None:
        log_path = pathlib.Path(config['OUTPUT_DIR']).joinpath(now_time.strftime('%y%m%d_%H%M%S'))
    if not log_path.exists():
        os.mkdir(str(log_path))

    json.dump(config, open(log_path.joinpath(config_path.name), mode='a'))

    # Load the dataset
    ds = dataset.SpectrogramDataSet(data_dir=config['DATA_DIR'],
                                    categories=config['CATEGORIES'], join_cat=config["CATEGORIES_TO_JOIN"],
                                    locations=config['LOCATIONS'],
                                    corrected=config['USE_CORRECTED_DATASET'],
                                    samples_per_class=config['SAMPLES_PER_CLASS'])

    # Define initial noise percentage
    noise_init_training = config['NOISE_RATIO'][0]
    noise_init_test = config['NOISE_RATIO_TEST'][0]

    scores = pd.DataFrame()
    con_matrix = pd.DataFrame()
    if type(config['TEST_SPLIT']) == float:
        print('Performing single train/validation/test split (random). Only one result will be given')
        paths_df = ds.prepare_all_dataset(test_size=config['TEST_SPLIT'], valid_size=config['VALID_SPLIT'],
                                          noise_ratio=noise_init_training)
        # Create and train the model
        scores_i, con_matrix_i = training.create_train_and_test_model(log_path, paths_df, config=config,
                                                                      fold=0, ds=ds)

        # scores.loc[0] = ['random', scores_i[0], scores_i[1]]
        scores = scores_i
        con_matrix = con_matrix_i.reset_index(drop=False, names='label')

    elif type(config['TEST_SPLIT']) == int:
        print('Performing K-fold stratified cross validation with K=%s. '
              'The cross validation is done to split TRAINING/VALIDATION vs TEST, '
              'the train-validation split is done randomly. This is for better error estimation. '
              'Results are given per fold' % config['TEST_SPLIT'])
        for fold, paths_df in ds.folds(noise_ratio=noise_init_training,
                                       n_folds=config['TEST_SPLIT'], valid_size=config['VALID_SPLIT']):
            # Create and train the model
            scores_i, con_matrix_i = training.create_and_train_model(log_path, paths_df, config=config,
                                                                     fold=fold, ds=ds)

            scores_i['fold'] = fold
            scores = pd.concat([scores, scores_i], ignore_index=True)
            con_matrix_i = con_matrix_i.reset_index(drop=False, names='label')
            con_matrix_i['fold'] = fold
            con_matrix = pd.concat([con_matrix, con_matrix_i], ignore_index=True)

    else:
        print('Performing blocked cross validation for each location (leave location out). '
              'Results are given per excluded location')
        for loc in config['LOCATIONS']:
            paths_df = ds.prepare_blocked_dataset(
                                                    valid_size=config[
                                                        'VALID_SPLIT'],
                                                    noise_ratio=noise_init_training,
                                                    blocked_location=loc,
                                                    noise_ratio_test=noise_init_test)
            # Create and train the model
            scores_i, con_matrix_i = training.create_train_and_test_model(log_path, paths_df, config=config,
                                                                          fold=loc, ds=ds)

            scores_i['fold'] = loc
            scores = pd.concat([scores, scores_i], ignore_index=True)
            con_matrix_i = con_matrix_i.reset_index(drop=False, names='label')
            con_matrix_i['fold'] = loc
            con_matrix = pd.concat([con_matrix, con_matrix_i], ignore_index=True)

    con_matrix.to_csv(log_path.joinpath('total_confusion_matrix.csv'))
    scores.to_csv(log_path.joinpath('total_scores.csv'))
    con_matrix = con_matrix.drop(columns=['fold'])
    con_matrix_avg = con_matrix.groupby('label').mean()
    model.plot_confusion_matrix(con_matrix_avg, save_path=log_path.joinpath('mean_confusion_matrix.png'))

    return scores, con_matrix


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define the parameters for the study
    config_file = input('Config file path:')
    if config_file == '':
        config_file = './config.json'
    run_from_config(pathlib.Path(config_file))
