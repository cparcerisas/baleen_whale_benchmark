import pandas as pd
import os

import model


def create_and_train_model(save_path, paths_df, ds, config, model_name):
    """
    Create and train model, from config

    :param log_path:
    :param paths_df:
    :param ds:
    :param config:
    :param fold:
    :return:
    """
    print(config['CATEGORIES'])
    m = model.Model(save_path=save_path, categories=config['CATEGORIES'], model_name=model_name)

    m.create(n_classes=ds.n_classes, batch_size=config['BATCH_SIZE'])

    x_train, y_train = ds.load_set_from_df(paths_df, 'train')
    x_valid, y_valid = ds.load_set_from_df(paths_df, 'valid')

    history = m.train(x_train, y_train, x_valid, y_valid, batch_size=config['BATCH_SIZE'], epochs=config['EPOCHS'],
                      loss_function=config['loss_function'], early_stop=config['early_stop'],
                      monitoring_metric=config['monitoring_metric'],
                      monitoring_direction=config['monitoring_direction'],
                      class_weights=config['CLASS_WEIGHTS'], learning_rate=config['learning_rate'])
    m.plot_training_metrics(history, chosen_metric=config['monitoring_metric'])
    m.save()
    paths_df.to_csv(m.log_path.joinpath('data_used_%s.csv' % model_name))
    return m


def run_multiple_models(log_path, paths_df, config, fold, ds, perform_test=False):
    """
    Create all the models with the specified noise on the training set
    train it and test it according to the specifications in config
    """
    if type(config['NOISE_RATIO']) == list:
        noise_to_train = config['NOISE_RATIO']
    else:
        noise_to_train = [config['NOISE_RATIO']]

    scores = pd.DataFrame()
    con_mat_df = pd.DataFrame()
    for i, noise in enumerate(noise_to_train):
        if i == 0:
            noise_before = noise
        else:
            noise_before = noise_to_train[i - 1]
        model_name = 'fold_%s_noise_%s' % (fold, noise)
        paths_df1, noise = select_more_noise(paths_df, 'train', noise_before, noise, config, ds)
        paths_df2, noise = select_more_noise(paths_df1, 'valid', noise_before, noise, config, ds)
        cnn_model = create_and_train_model(log_path, paths_df2, ds, config, model_name=model_name)
        if perform_test:
            scores_i, con_mat_i = test_model_multiple_noise(cnn_model, paths_df, config, ds, fold, log_path)
            scores = pd.concat([scores, scores_i])
            con_mat_df = pd.concat([con_mat_df, con_mat_i])

    return scores, con_mat_df


def test_model_multiple_noise(cnn_model, paths_df, config, ds, fold, log_path):
    if type(config['NOISE_RATIO_TEST']) == list:
        noise_to_test = config['NOISE_RATIO_TEST']
    else:
        noise_to_test = [config['NOISE_RATIO_TEST']]

    last_noise = noise_to_test[0]
    scores_i = pd.DataFrame()
    con_mat_df = pd.DataFrame()
    for noise_test in noise_to_test:
        paths_df, train_noise = select_more_noise(paths_df, 'test', last_noise, noise_test, config, ds)
        last_noise = noise_test
        scores_noise, con_mat_noise, predictions = cnn_model.test_in_batches(ds, data_split_df=paths_df)

        model.plot_confusion_matrix(con_mat_noise, log_path.joinpath('confusion_matrix_fold%s_noise%s_noise%s.png' %
                                                                     (fold, train_noise, noise_test)))
        # Add the metadata
        scores_noise['noise_percentage_train'] = train_noise
        scores_noise['noise_percentage_test'] = noise_test

        scores_i = pd.concat([scores_i, scores_noise])
        con_mat_df = pd.concat([con_mat_df, con_mat_noise])

        paths_df.to_csv(
            log_path.joinpath('data_used_fold%s_noise%s_noise%s.csv' % (fold, train_noise, noise_test)))
        predictions.to_csv(
            log_path.joinpath('predictions_fold%s_noise%s_noise%s.csv' % (fold, train_noise, noise_test)))

    return scores_i, con_mat_df


def test_model_from_folder(folder_path, ds):
    m = model.Model(save_path=folder_path.parent, categories=ds.categories, model_name=folder_path.name)
    m.load_existing()
    csv_split_file = m.log_path.joinpath('data_used_%s.csv' % m.model_name)
    con_mat = m.predict_full_ds(ds, csv_split_file)
    return con_mat


def select_more_noise(paths_df, phase, noise, new_noise, config, ds):
    if noise == new_noise:
        return paths_df, noise
    elif new_noise == 'all':
        paths_df = ds.select_more_noise(paths_df, new_noise, phase)
    elif noise != 'all':
        if phase == 'test':
            if new_noise > noise:
                paths_df = ds.select_more_noise(paths_df, new_noise, phase)
            elif new_noise < config['NOISE_RATIO'][0]:
                print(
                    'Noise percentage lower than in first training. '
                    'Not considering it and testing on the first training ratio'
                )
                noise2 = config['NOISE_RATIO'][0]

        else:
            if new_noise > noise:
                paths_df = ds.select_more_noise(paths_df, new_noise, phase)

    return paths_df, noise2
