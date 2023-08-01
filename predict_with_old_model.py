import pathlib
import json
import cv2
import os

import pandas as pd
import tensorflow as tf
import numpy as np

import dataset
import training
import metrics

data_folder = pathlib.Path('/Users/eschall/Documents/Python/CNN+Noise/Specs')
model_to_analyze_folder = pathlib.Path('/Users/eschall/Documents/Python/CNN+Noise/Results/230731_152252/foldBallenyIsland2015/model')
config_folder = pathlib.Path('/Users/eschall/Documents/Python/CNN+Noise/Results/230731_152252')
csv_split_file = pathlib.Path('/Users/eschall/Documents/Python/CNN+Noise/Results/230731_152252/data_used_foldBallenyIsland2015_noiseall.csv')

IMAGE_HEIGHT = 90
IMAGE_WIDTH = 30
N_CHANNELS = 1

class SpectrogramDataSet:
    def __init__(self,  image_width, image_height, categories, join_cat, n_channels):
        """

        :param image_width:
        :param image_height:
        :param categories:
        :param join_cat:

        """

        self.image_height = image_height
        self.image_width = image_width
        self.categories = categories
        self.join_cat = join_cat
        self.n_channels = n_channels

        # Create an understandable map for joined categories
        # and their corresponding int representation
        self.map_join = {}
        self.classes2int = {}
        self.int2class = []
        for join_class_name, classes_list in join_cat.items():
            self.classes2int[join_class_name] = len(self.classes2int.keys())
            self.int2class.append(join_class_name)
            for class_name in classes_list:
                self.map_join[class_name] = join_class_name

        for cat_name in self.categories:
            if cat_name not in self.map_join.keys():
                self.map_join[cat_name] = cat_name
                self.classes2int[cat_name] = len(self.classes2int)
                self.int2class.append(cat_name)

    def reshape_images(self, images):
        """
        Reshape all the images to the specified height, width and channels during init of object

        :param images: list of images
        :return: array with normalized images (0 to 1) with the correct shape
        """
        X = np.array(images).reshape(-1, self.image_width, self.image_height, self.n_channels)
        x = X / 255.0
        return x

def predict_model_full_dataset(data_folder, model_folder, config_folder, csv_split_file):
    #  Load the test dataset
    x_test, y_test, ds = load_dataset_from_csv(csv_split_file=csv_split_file, data_folder=data_folder, config_folder=config_folder)

    detection_metrics = metrics.ImbalancedDetectionMatrix(noise_class_name='Noise', classes_names=ds.int2class)
    cnn_model = tf.keras.models.load_model(model_folder, custom_objects={"imbalanced_metric": detection_metrics.imbalanced_metric, "noise_misclas_rate": detection_metrics.noise_misclas_rate, "call_avg_tpr": detection_metrics.call_avg_tpr, "f1_score": metrics.f1_score, "recall_score": metrics.recall_score, "accuracy":tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')},)

    scores_i, con_mat_df = training.get_model_scores(cnn_model, x_test=x_test, y_test=y_test, categories=ds.int2class)


    return scores_i, con_mat_df


def load_dataset_from_csv(csv_split_file, data_folder, config_folder):

    config_file = config_folder.joinpath('config.json')

    # Read the config file
    f = open(config_file)
    config = json.load(f)

    ds = SpectrogramDataSet(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                        categories=config['CATEGORIES'], join_cat=config["CATEGORIES_TO_JOIN"],
                        n_channels=N_CHANNELS)

    data_split_df = pd.read_csv(csv_split_file)
    data_split_test = data_split_df[data_split_df['set'] == 'test']
    images_for_test = pd.Series(data_split_test['path'])

    labels = []
    images = []
    for img_path in images_for_test:
        cat_folder = os.path.splitext(os.path.basename(img_path))[0].split('_')[2]
        img_array = cv2.imread(os.path.join(data_folder, cat_folder, img_path))

        # Not necessary if images already on the correct format
        # resized_image = cv2.resize(img_array, (self.image_width, self.image_height))
        grey_image = np.mean(img_array, axis=2)
        images.append(grey_image)

        # This part is for joined classes
        labels.append(ds.classes2int[ds.map_join[cat_folder]])

    x = ds.reshape_images(images)
    y = np.array(labels)

    return x, y, ds


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Test an old model
    scores, con_matrix = predict_model_full_dataset(data_folder=data_folder, model_folder=model_to_analyze_folder, config_folder=config_folder, csv_split_file=csv_split_file)

    log_path = pathlib.Path(os.path.split(model_to_analyze_folder)[0])
    con_matrix.to_csv(log_path.joinpath('confusion_matrix.csv'))
    scores.to_csv(log_path.joinpath('scores.csv'))

