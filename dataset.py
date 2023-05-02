import os
import collections
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43


class SpectrogramDataSet:
    def __init__(self, data_dir, image_width, image_height, categories, join_cat, locations, n_channels, corrected):
        self.locations = locations
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.categories = categories
        self.corrected = corrected
        self.join_cat = join_cat

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

        self.n_classes = len(self.classes2int)
        print('These are the classes: ', self.classes2int)
        print('Which are formed by doing: ', self.map_join)

    def reshape_images(self, images):
        """
        Reshape all the images to the specified height, width and channels during init of object

        :param images: list of images
        :return: array with normalized images (0 to 1) with the correct shape
        """
        X = np.array(images).reshape(-1, self.image_width, self.image_height, self.n_channels)
        x = X / 255.0
        return x

    def load_data(self, samples_per_class, noise_ratio, locations_to_exclude=None):
        """
        The function will return the selected data for all the classes together, shuffled.
        For non-noise classes, the data included will be the first samples_per_class of the dataset (ordered),
        excluding the ones corresponding to the locations to exclude, if any.

        :param samples_per_class: int, number of samples per class
        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :param locations_to_exclude: list of locations to not load (for blocked testing)
        :return: x, y and paths
        """
        total_images = []
        total_labels = []
        total_paths = []

        # Loop through all the categories
        for cat_i, category in enumerate(self.categories):
            # If Noise, select a random amount
            if category == 'Noise':
                samples_per_class = self.get_noise_samples(samples_per_class, noise_ratio)

            # Add the data from that category
            images, labels, paths_list = self.load_data_category(category, samples_per_class, locations_to_exclude)
            total_images += images
            total_labels += labels
            total_paths += paths_list

        x = self.reshape_images(total_images)
        y = np.array(total_labels)
        x, y, total_paths = shuffle(x, y, total_paths)

        return x, y, total_paths

    def load_data_category(self, category, samples_per_class, locations_to_exclude=None,
                           samples_to_exclude=None):
        """
        The function will return the selected data.
        For non-noise classes, the data included will be the first samples_per_class of the dataset (ordered),
        excluding the ones listed in samples_to_exclude and the ones corresponding to the locations to exclude,
        if any.

        :param category: string, category to load
        :param samples_per_class: int, number of samples per class
        :param locations_to_exclude:
        :param samples_to_exclude:
        :return:
        """
        paths_list = []
        labels = []
        images = []
        path = os.path.join(self.data_dir, category)

        # Read all the images of that category
        all_images = pd.Series(os.listdir(path))

        # If there are one or more locations to exclude, exclude them from the list!
        if locations_to_exclude is not None:
            for loc in locations_to_exclude:
                all_images = all_images.loc[~all_images.str.contains(loc)]

        if samples_to_exclude is not None:
            all_images = all_images.loc[~all_images.isin(samples_to_exclude)]

        # If the dataset is corrected, exclude the corrections
        joined_cat = self.map_join[category]

        if samples_per_class == 'all':
            last_img = -1
        else:
            # Sort the images, we only want the n first ones
            order = all_images.str.split('_', expand=True)[0].astype(int)
            order = order.sort_values()
            all_images = all_images.reindex(order.index)
            last_img = min(len(all_images), int(samples_per_class))

        selected_images = all_images.iloc[:last_img]
        # If using the corrected dataset, eliminate the ones that are not correct
        if self.corrected and category != 'Noise':
            correction_path = os.path.join(self.data_dir, category + '2Noise.csv')
            if not os.path.exists(correction_path):
                raise Exception('If you want to use the corrected dataset you should provide a csv file with the '
                                'corrections for each of the original classes')
            correction_csv = pd.read_csv(correction_path, header=None)
            all_images_joined_names = selected_images.str.split('_').str.join('')
            selected_images = selected_images.loc[~all_images_joined_names.isin(correction_csv[0])]

        for img_path in selected_images:
            img_array = cv2.imread(os.path.join(path, img_path))

            # Not necessary if images already on the correct format
            # resized_image = cv2.resize(img_array, (self.image_width, self.image_height))
            grey_image = np.mean(img_array, axis=2)
            images.append(grey_image)

            # This part is for joined classes
            labels.append(self.classes2int[joined_cat])
            paths_list.append(img_path)

        return images, labels, paths_list

    def load_all_dataset(self, test_size, valid_size, samples_per_class, noise_ratio):
        """
        Will load all the labeled data (up to samples_per_class for each class)
        and some noise samples up to a certain noise_ratio

        :param test_size: float (0 to 1), percentage from the total data loaded to split randomly to test
        :param valid_size: float (0 to 1), percentage from the model (not test) data loaded to split randomly to
        validation
        :param samples_per_class: int, number of samples per class
        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :return: x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list (of all the data)
        """
        x, y, paths_list = self.load_data(locations_to_exclude=None, samples_per_class=samples_per_class,
                                          noise_ratio=noise_ratio)
        x_model, x_test, y_model, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
        x_train, x_valid, y_train, y_valid = train_test_split(x_model, y_model, test_size=valid_size, shuffle=True)
        return x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list

    def load_blocked_dataset(self, blocked_location, valid_size, samples_per_class, noise_ratio):
        """
        Same than load_all_dataset but the test is decided by the blocked location
        :param blocked_location: string, name of the location to use for test and NOT for training or validation
        :param valid_size: float (0 to 1), percentage from the model (not test) data loaded to split randomly to
        validation
        :param samples_per_class: int, number of samples per class
        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :return: x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list (of all the data)
        """
        selected_locs = list(set(self.locations) - {blocked_location})
        x, y, paths_list_model = self.load_data(locations_to_exclude=[blocked_location],
                                                samples_per_class=samples_per_class,
                                                noise_ratio=noise_ratio)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size, shuffle=True)

        x_test, y_test, paths_list_test = self.load_data(locations_to_exclude=selected_locs,
                                                         samples_per_class=samples_per_class, noise_ratio=noise_ratio)

        paths_list = paths_list_model + paths_list_test
        return x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list

    def folds(self, samples_per_class, noise_ratio, n_folds, valid_size):
        """
        Loop through the folds. The data will be first loaded (x, y) according to the samples per class and noise ratio.
        Once the data is loaded, it will be split between model and test according to the folds.
        Inside every fold, model split is split further into train and test, but this time randomly.

        :param samples_per_class: int, number of samples per class
        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :param n_folds: number of folds to loop through
        :param valid_size: float (0 to 1) validation split from the model split
        :return: fold, x_train, y_train, x_vali, y_valid, x_test, y_test, paths_list (for all together)
        """
        x, y, paths_list = self.load_data(samples_per_class,
                                          noise_ratio=noise_ratio,
                                          locations_to_exclude=None)
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
        for fold, (train_index, test_index) in enumerate(kfold.split(x, y)):
            x_model = x[train_index]
            y_model = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
            x_train, x_valid, y_train, y_valid = train_test_split(x_model, y_model,
                                                                  test_size=valid_size, shuffle=True)
            yield fold, x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list

    def load_more_noise(self, x_test, y_test, paths_list, new_noise_ratio, samples_per_class):
        """
        Append to x_test and y_test more noise, NOT repeated (not the same samples).
        The amount of noise added is according to the new_noise_ratio.

        :param x_test: existing x_test
        :param y_test: existing y_test
        :param paths_list: paths of the images corresponding to x_test and y_test
        :param new_noise_ratio: new ratio (0 to 1) of noise from the total dataset
        :param samples_per_class: number of samples per class
        :return: updated x_test and y_test
        """
        noise_samples = self.get_noise_samples(samples_per_class, new_noise_ratio)
        new_noise_samples = noise_samples - (y_test == self.classes2int['Noise']).sum()
        images, labels, new_paths_list = self.load_data_category('Noise', new_noise_samples, locations_to_exclude=None,
                                                                 samples_to_exclude=paths_list)

        new_x = self.reshape_images(images)
        new_y = np.array(labels)

        x = np.concatenate([x_test, new_x])
        y = np.concatenate([y_test, new_y])

        paths_list += new_paths_list

        return x, y, paths_list

    def get_noise_samples(self, samples_per_class, noise_ratio):
        """
        Compute how many noise samples are necessary to get the specified noise_ratio if each class has an amount of
        samples of samples_per_class

        :param samples_per_class: int, number of samples per class
        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :return: number of samples
        """
        return ((len(self.categories) - 1) * samples_per_class * noise_ratio) / (1 - noise_ratio)
