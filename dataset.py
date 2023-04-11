import os
import collections
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43


class SpectrogramDataSet:
    def __init__(self, data_dir, image_width, image_height, categories, join_cat, locations,  n_channels, corrected):
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
        for join_class_name, classes_list in join_cat.items():
            self.classes2int[join_class_name] = len(self.classes2int.keys())
            for class_name in classes_list:
                self.map_join[class_name] = join_class_name

        for cat_name in self.categories:
            if cat_name not in self.map_join.keys():
                self.map_join[cat_name] = cat_name
                self.classes2int[join_class_name] = len(self.classes2int)

        self.n_classes = len(self.classes2int)
        print('These are the classes: ', self.classes2int)
        print('Which are formed by doing: ', self.map_join)

    def _load_data(self, samples_per_class, noise_ratio, locations_to_exclude=None):
        paths_list = []
        labels = []
        images = []
        # Loop through all the categories
                
        for cat_i, category in enumerate(self.categories):
            path = os.path.join(self.data_dir, category)

            # Read all the images of that category
            all_images = pd.Series(os.listdir(path))
            
            # If there are one or more locations to exclude, exclude them from the list!
            if locations_to_exclude is not None:
                for loc in locations_to_exclude:
                    all_images = all_images.loc[~all_images.str.contains(loc)]

            # If the dataset is corrected, exclude the corrections
            joined_cat = self.map_join[category]

            # If using the corrected dataset, eliminate the ones that are not correct
            if self.corrected:
                correction_path = os.path.join(self.data_dir, category + '2Noise.csv')
                if os.path.exists(correction_path):
                    correction_csv = pd.read_csv(correction_path, header=None)
                    all_images_joined_names = all_images.str.split('_').str.join('')
                    all_images = all_images.loc[all_images_joined_names.isin(correction_csv[0])]

            # Select a random amount
            if category == 'Noise':
                samples_per_class = ((len(self.categories) - 1) * samples_per_class * noise_ratio) / (1 - noise_ratio)

            if samples_per_class == 'all':
                last_img = -1
            else:
                # Sort the images, we only want the n first ones
                order = all_images.str.split('_', expand=True)[0].astype(int)
                order = order.sort_values()
                all_images = all_images.reindex(order.index)
                last_img = min(len(all_images), int(samples_per_class))

            selected_images = all_images.iloc[:last_img]
            for img_path in selected_images:
                img_array = cv2.imread(os.path.join(path, img_path))

                # Not necessary if images already on the correct format
                # resized_image = cv2.resize(img_array, (self.image_width, self.image_height))
                grey_image = np.mean(img_array, axis=2)
                images.append(grey_image)

                # This part is for joined classes
                labels.append(self.classes2int[joined_cat])
                paths_list.append(img_path)
        X = np.array(images).reshape(-1, self.image_width, self.image_height, self.n_channels)
        x = X / 255.0
        y = np.array(labels)

        x, y, paths_list = shuffle(x, y, paths_list)

        return x, y

    def load_all_dataset(self, test_size, valid_size, samples_per_class, noise_ratio):
        x, y = self._load_data(locations_to_exclude=None, samples_per_class=samples_per_class, noise_ratio=noise_ratio)
        x_model, x_test, y_model, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
        x_train, x_valid, y_train, y_valid = train_test_split(x_model, y_model, test_size=valid_size, shuffle=True)
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def load_blocked_dataset(self, blocked_location, valid_size, samples_per_class, noise_ratio):
        selected_locs = list(set(self.locations) - {blocked_location})
        x, y = self._load_data(locations_to_exclude=[blocked_location],
                               samples_per_class=samples_per_class,
                               noise_ratio=noise_ratio)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size, shuffle=True)

        x_test, y_test = self._load_data(locations_to_exclude=selected_locs,
                                         samples_per_class=samples_per_class, noise_ratio=noise_ratio)

        return x_train, y_train, x_valid, y_valid, x_test, y_test
