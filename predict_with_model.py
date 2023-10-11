import pathlib
import json
import os

import dataset
import training
import model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_folder = pathlib.Path(r"D:\data\Miller_AntarcticData\Specs_sequenced")
    model_folder = pathlib.Path(
        r"D:\data\Miller_AntarcticData\CNN+Noise_Results\230821_094657\foldMaudRise2014\model")
    config_folder = pathlib.Path(r"D:\data\Miller_AntarcticData\CNN+Noise_Results\230821_094657")
    from_csv = False
    csv_split_file = pathlib.Path(
        r"D:\data\Miller_AntarcticData\CNN+Noise_Results\230801_113332_BallenyOut\
        data_used_foldRossSea2014_noiseall.csv")

    config_file = config_folder.joinpath('config.json')

    # Read the config file
    f = open(config_file)
    config = json.load(f)

    #  Load the test dataset
    ds = dataset.SpectrogramDataSet(data_dir=config['DATA_DIR'],
                                    categories=config['CATEGORIES'], join_cat=config["CATEGORIES_TO_JOIN"],
                                    locations=config['LOCATIONS'],
                                    corrected=config['USE_CORRECTED_DATASET'],
                                    samples_per_class=config['SAMPLES_PER_CLASS'])
    m = model.Model(log_path=model_folder, categories=ds.categories)
    # Test an old model
    con_matrix = m.predict_full_ds(data_folder=data_folder, model_folder=model_folder,
                                   config_folder=config_folder, csv_split_file=csv_split_file)

    log_path = pathlib.Path(os.path.split(model_folder)[0])
    con_matrix.to_csv(log_path.joinpath('confusion_matrix.csv'))
    # scores.to_csv(log_path.joinpath('scores.csv'))
