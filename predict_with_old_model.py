import pathlib
import json

import pandas as pd
import tensorflow as tf
from sklearn.metrics import RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt
import numpy as np

import dataset


def predict_model_full_dataset(data_folder, model_folder):
    config_file = model_folder.joinpath('config.json')

    # Read the config file
    f = open(config_file)
    config = json.load(f)

    rois_path = data_folder.joinpath(config['data_file'])
    rois_df = pd.read_csv(rois_path)

    #  Create the dataset
    ds = dataset.AudioDataSetROI(data_folder, rois_df, fs=config['SAMPLING_RATE'], block_time=config['BLOCK_TIME'],
                                 pad_time=config['PAD_TIME'])

    model = tf.keras.models.load_model(model_folder.joinpath('model'))

    ds.predict_with_model(model, output_folder=model_folder,
                          threshold=config['threshold_results'])


def plot_roc_curve(csv_path_with_prob):
    prob_csv = pd.read_csv(csv_path_with_prob)
    prob_csv = prob_csv.dropna()
    y_test = prob_csv.Label != 'Noise'
    y_pred = prob_csv['prob']

    fpr, tpr, thresholds_list = roc_curve(y_test, y_pred)
    fpr_1percent = np.argmin(np.abs(fpr - 0.01))
    tpr_1percernt = tpr[fpr_1percent]
    threshold = thresholds_list[fpr_1percent]
    print('TPR for 1% FPR: ', tpr_1percernt)
    print('Treshold where that happens: ', threshold)
    RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.savefig(csv_path_with_prob.parent.joinpath('ROC_curve.png'))
    plt.show()

    roc_df = pd.DataFrame(data={'fpr': fpr, 'tpr': tpr, 'threshold': thresholds_list})
    roc_df.to_csv(csv_path_with_prob.parent.joinpath('ROC_values.csv'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_folder = pathlib.Path('./data')
    models_folder = pathlib.Path('./transfer_models')

    model_to_analyze_folder = models_folder.joinpath('transfer_mammals', '220727_170706')
    # Test an old model
    predict_model_full_dataset(data_folder=data_folder, model_folder=model_to_analyze_folder)
    plot_roc_curve(model_to_analyze_folder.joinpath('probability_predictions.csv'))
