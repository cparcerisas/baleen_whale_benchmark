import pandas as pd
import os

import model


def create_and_train_model(log_path, paths_df, ds, config, fold):
    """
    Create and train model, from config

    :param log_path:
    :param paths_df:
    :param ds:
    :param config:
    :param fold:
    :return:
    """
    model_log_path = log_path.joinpath('fold%s' % fold)
    if not model_log_path.exists():
        os.mkdir(str(model_log_path))
    print(config['CATEGORIES'])
    m = model.Model(log_path=model_log_path, categories=config['CATEGORIES'])
    m.create(n_classes=ds.n_classes, batch_size=config['BATCH_SIZE'])

    x_train, y_train = ds.load_set_from_df(paths_df, 'train')
    x_valid, y_valid = ds.load_set_from_df(paths_df, 'valid')

    history = m.train(x_train, y_train, x_valid, y_valid, batch_size=config['BATCH_SIZE'], epochs=config['EPOCHS'],
                      loss_function=config['loss_function'], early_stop=config['early_stop'],
                      monitoring_metric=config['monitoring_metric'],
                      monitoring_direction=config['monitoring_direction'],
                      class_weights=config['CLASS_WEIGHTS'], learning_rate=config['learning_rate'])

    m.plot_training_metrics(history, fold=fold, chosen_metric=config['monitoring_metric'])
    m.save(extra_info=fold)
    return m


def create_train_and_test_model(log_path, paths_df, config, fold, ds):
    """
    Create the model, train it and test it according to the specifications in config
    """
    if type(config['NOISE_RATIO']) == list:
        noise_to_train = config['NOISE_RATIO']
    else:
        noise_to_train = [config['NOISE_RATIO']]

    scores_i = pd.DataFrame()
    con_mat_df = pd.DataFrame()

    for i, noise in enumerate(noise_to_train):
        if i == 0:
            noise_before = noise
        else:
            noise_before = noise_to_train[i - 1]
        paths_df1, noise = select_more_noise(paths_df, 'train', noise_before, noise, config, ds)
        paths_df2, noise = select_more_noise(paths_df1, 'valid', noise_before, noise, config, ds)
        create_and_train_model(log_path, paths_df2, ds, config, fold)

    return scores_i, con_mat_df


def test_model_multiple_noise(cnn_model, paths_df, config, ds, noise_percentage_train, fold, log_path):
    if type(config['NOISE_RATIO_TEST']) == list:
        noise_to_test = config['NOISE_RATIO_TEST']
    else:
        noise_to_test = [config['NOISE_RATIO_TEST']]
    x_test = None
    y_test = None
    for noise_test in noise_to_test:
        if x_test is None:
            x_test, y_test = ds.load_test()
        else:
            paths_df, noise = select_more_noise(paths_df, 'test', noise_percentage_train, noise_test, config, ds)

        ds.load_set_from_df(paths_df, 'test')

        scores_noise, con_mat_noise, predictions = cnn_model.test(x_test, y_test, ds.categories,
                                                                  paths_df.path[paths_df.set == 'test'])
        scores_noise['noise_percentage_train'] = noise_percentage_train
        con_mat_noise['noise_percentage_train'] = noise_percentage_train
        scores_noise['noise_percentage_test'] = noise_test
        con_mat_noise['noise_percentage_test'] = noise_test
        scores_i = pd.concat([scores_i, scores_noise])
        con_mat_df = pd.concat([con_mat_df, con_mat_noise])
        model.plot_confusion_matrix(con_mat_noise, log_path.joinpath('confusion_matrix_fold%s_noise%s_noise%s.png' %
                                                                         (fold, noise_percentage_train, noise_test)))
        paths_df.to_csv(
            log_path.joinpath('data_used_fold%s_noise%s_noise%s.csv' % (fold, noise_percentage_train, noise_test)))
        predictions.to_csv(
            log_path.joinpath('predictions_fold%s_noise%s_noise%s.csv' % (fold, noise_percentage_train, noise_test)))


def select_more_noise(paths_df, phase, noise, noise2, config, ds):
    if noise != 'all':
        if phase == 'test':
            if noise2 > config['NOISE_RATIO'][0]:
                paths_df = ds.select_more_noise(paths_df, noise2, phase)
            elif noise2 < config['NOISE_RATIO'][0]:
                print(
                    'Noise percentage lower than in first training. '
                    'Not considering it and testing on the first training ratio'
                )
                noise2 = config['NOISE_RATIO'][0]

        else:
            if noise2 > noise:
                paths_df = ds.select_more_noise(paths_df, noise2, phase)

    return paths_df, noise2
