import json

import pandas as pd

import dataset
import training
import model


def test_multiple_models(mother_folder, ds):
    con_mat = pd.DataFrame()
    for folder in mother_folder.glob('*'):
        if folder.isdir():
            con_mat_i = training.test_model_from_folder(folder, ds)
            con_matrix_i = con_matrix_i.reset_index(drop=False, names='label')
            con_matrix_i['fold'] = folder.name
            con_mat = pd.concat([con_mat, con_mat_i])
    con_matrix = con_mat.drop(columns=['fold'])
    con_matrix_avg = con_matrix.groupby('label').mean()
    model.plot_confusion_matrix(con_matrix_avg, save_path=mother_folder.joinpath('mean_confusion_matrix.png'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config_file = input('Where is the config file?')
    # Read the config file
    f = open(config_file)
    config = json.load(f)

    model_folder = input('Where is the folder to test?')

    #  Load the test dataset
    spectro_ds = dataset.SpectrogramDataSet(data_dir=config['DATA_DIR'],
                                            categories=config['CATEGORIES'], join_cat=config["CATEGORIES_TO_JOIN"],
                                            locations=config['LOCATIONS'],
                                            corrected=config['USE_CORRECTED_DATASET'],
                                            samples_per_class=config['SAMPLES_PER_CLASS'])

    test_multiple_models(model_folder, spectro_ds)
