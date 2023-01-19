import os

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43


class SpectrogramDataSet:
    def __init__(self, data_dir, image_width, image_height, categories, locations, n_channels):
        self.locations = locations
        self.categories = categories
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels

    def _load_data(self, samples_per_class, noise_ratio, locations_to_exclude=None):
        paths_list = []
        labels = []
        images = []
        # Loop through all the categories
        for cat_i, category in enumerate(self.categories):
            path = os.path.join(self.data_dir, category)

            # Read all the images of that category
            all_images = pd.Series(os.listdir(path))
            if category == 'Noise':
                samples_per_class = ((len(self.categories) - 1) * samples_per_class * noise_ratio) / (1 - noise_ratio)

            # If there are one or more locations to exclude, exclude them from the list!
            if locations_to_exclude is not None:
                for loc in locations_to_exclude:
                    all_images = all_images.loc[~all_images.str.contains(loc)]

            # Select a random amount
            selected_images = all_images.sample(min(len(all_images), int(samples_per_class)))
            for img_path in selected_images:
                img_array = cv2.imread(os.path.join(path, img_path))

                resized_image = cv2.resize(img_array, (self.image_width, self.image_height))
                grey_image = np.mean(resized_image, axis=2)
                images.append(grey_image)
                labels.append(cat_i)
                paths_list.append(img_path)

        X = np.array(images).reshape(-1, self.image_width, self.image_height, self.n_channels)
        x = X / 255.0
        y = np.array(labels)

        x, y, paths_list = shuffle(x, y, paths_list)

        return x, y

    def load_all_dataset(self, test_size, valid_size, samples_per_class, noise_ratio):
        x, y = self._load_data(locations=self.locations, samples_per_class=samples_per_class, noise_ratio=noise_ratio)
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
