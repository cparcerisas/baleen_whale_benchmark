import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 42

IMAGE_HEIGHT = 90
IMAGE_WIDTH = 30
N_CHANNELS = 1

CORRECTED_SAMPLES = 2500


class SpectrogramDataSet:
    def __init__(self, data_dir, categories, join_cat, locations, corrected,
                 samples_per_class='all'):
        """
        :param data_dir:
        :param categories:
        :param join_cat:
        :param locations:
        :param n_channels:
        :param corrected:
        :param samples_per_class: int, number of samples per class

        """
        self.locations = locations
        self.data_dir = data_dir
        self.categories = categories
        self.corrected = corrected
        self.join_cat = join_cat
        self.samples_per_class = samples_per_class

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

    def how_many_samples(self):
        """
        Return the number of samples if all is selected
        :return:
        """
        samples = 0
        for cat_i, category in enumerate(self.categories):
            # If Noise, select a random amount
            if category != 'Noise':
                path = os.path.join(self.data_dir, category)

                # Read all the images of that category
                samples += len(pd.Series(os.listdir(path)))
        return samples / (len(self.int2class) - 1)

    @staticmethod
    def reshape_images(images):
        """
        Reshape all the images to the specified height, width and channels during init of object

        :param images: list of images
        :return: array with normalized images (0 to 1) with the correct shape
        """
        X = np.array(images).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, N_CHANNELS)
        x = X / 255.0
        return x

    @staticmethod
    def join_paths_to_df(paths_train, paths_valid, paths_test):
        train_df = pd.DataFrame({'path': paths_train})
        valid_df = pd.DataFrame({'path': paths_valid})
        test_df = pd.DataFrame({'path': paths_test})
        train_df = train_df.assign(set='train')
        valid_df = valid_df.assign(set='valid')
        test_df = test_df.assign(set='test')
        paths_df = pd.concat([train_df, valid_df, test_df])
        return paths_df

    def prepare_all_dataset(self, test_size, valid_size, noise_ratio):
        """
        Will load all the labeled data (up to samples_per_class for each class)
        and some noise samples up to a certain noise_ratio

        :param test_size: float (0 to 1), percentage from the total data loaded to split randomly to test
        :param valid_size: float (0 to 1), percentage from the model (not test) data loaded to split randomly to
        validation
        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :return: x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list (of all the data)
        """
        paths_list = self.select_data(locations_to_exclude=None, noise_ratio=noise_ratio)
        x, y = self.load_from_file_list(file_list=paths_list)
        x_model, x_test, y_model, y_test, paths_model, paths_test = train_test_split(x, y, paths_list,
                                                                                     test_size=test_size, shuffle=True)
        x_train, x_valid, y_train, y_valid, paths_train, paths_valid = train_test_split(x_model, y_model, paths_model,
                                                                                        test_size=valid_size,
                                                                                        shuffle=True)
        paths_df = self.join_paths_to_df(paths_train, paths_valid, paths_test)

        return paths_df

    def prepare_blocked_dataset(self, blocked_location, valid_size, noise_ratio, noise_ratio_test):
        """
        Same than prepare_all_dataset but the test is decided by the blocked location
        :param blocked_location: string, name of the location to use for test and NOT for training or validation
        :param valid_size: float (0 to 1), percentage from the model (not test) data loaded to split randomly to
        validation
        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :return: x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list (of all the data)
        """
        selected_locs = list(set(self.locations) - {blocked_location})
        print('selecting model data...')
        paths_list_model = self.select_data(locations_to_exclude=[blocked_location], noise_ratio=noise_ratio)
        y = self.read_labels_from_file_list(file_list=paths_list_model)
        paths_train, paths_valid = train_test_split(paths_list_model, stratify=y, test_size=valid_size, shuffle=True)

        print('selecting test data...')
        paths_test = self.select_data(locations_to_exclude=selected_locs, noise_ratio=noise_ratio_test)
        paths_df = self.join_paths_to_df(paths_train, paths_valid, paths_test)
        return paths_df

    def folds(self, noise_ratio, n_folds, valid_size):
        """
        Loop through the folds. The data will be first loaded (x, y) according to the samples per class and noise ratio.
        Once the data is loaded, it will be split between model and test according to the folds.
        Inside every fold, model split is split further into train and test, but this time randomly.

        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :param n_folds: number of folds to loop through
        :param valid_size: float (0 to 1) validation split from the model split
        :return: fold, x_train, y_train, x_vali, y_valid, x_test, y_test, paths_list (for all together)
        """
        paths_list = self.select_data(noise_ratio=noise_ratio, locations_to_exclude=None)
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
        y = self.read_labels_from_file_list(file_list=paths_list)
        for fold, (train_index, test_index) in enumerate(kfold.split(paths_list, y)):
            paths_model = paths_list[train_index]
            y_model = y[train_index]
            paths_test = paths_list[test_index]

            paths_train, paths_valid = train_test_split(paths_model, stratify=y_model,
                                                        test_size=valid_size, shuffle=True)
            paths_df = self.join_paths_to_df(paths_train, paths_valid, paths_test)
            yield fold, paths_df

    def select_files_category(self, category, samples_to_load, locations_to_exclude=None,
                              samples_to_exclude=None):
        """
        The function will return the selected data.
        For non-noise classes, the data included will be the first samples_to_load of the dataset (ordered),
        excluding the ones listed in samples_to_exclude and the ones corresponding to the locations to exclude,
        if any.

        :param category: string, category to load
        :param samples_to_load: int, number of samples per category (sum of all subclasses)
        :param locations_to_exclude:
        :param samples_to_exclude:
        :return:
        """
        subcats = dict((key, value) for (key, value) in self.map_join.items() if value == category)
        # Load all the available samples from the folder
        num_samples = []
        for subcat in subcats:
            path = os.path.join(self.data_dir, subcat)
            num_samples.append(len(os.listdir(path)))
        num_samples = np.array(num_samples)

        if samples_to_load != 'all':
            n_subcategories = len(subcats)
            samples_per_subcategory = round(samples_to_load / n_subcategories)
            # The sum of all the samples of the subclasses is smaller than the samples to load
            if num_samples.sum() < samples_to_load:
                raise Exception('Samples per class too high for available dataset, please choose a lower number')

            # There is enough data at each subclass
            elif all(num_samples >= samples_per_subcategory):
                samples_to_load_per_subcat = np.repeat(samples_per_subcategory, n_subcategories)

            # If there is one of the subclasses which has less than its proportional part, check if we can load more
            # of the other subclasses, and do it iteratively until all the samples are reached
            elif any(num_samples < samples_per_subcategory):
                samples_df = pd.DataFrame(index=subcats.keys(), columns=['samples_to_load', 'available', 'needed'])
                samples_df['available'] = num_samples
                samples_df['needed'] = samples_per_subcategory
                samples_df['samples_to_load'] = samples_df[['available', 'needed']].min()
                total_loaded = samples_df.samples_to_load.sum()
                while total_loaded < samples_to_load:
                    leftover = samples_to_load - total_loaded
                    samples_df['needed'] += round(leftover / (samples_df['available'] > samples_df['needed']).sum())
                    samples_df['samples_to_load'] = samples_df[['available', 'needed']].min(axis=1)
                    total_loaded = samples_df.samples_to_load.sum()
                samples_to_load_per_subcat = samples_df['samples_to_load'].values
        else:
            samples_to_load_per_subcat = num_samples

        print(subcats, samples_to_load, num_samples)

        total_selected_subcat = []
        for i, subcat in enumerate(subcats):
            path = os.path.join(self.data_dir, subcat)

            # Read all the images of that category
            images_per_subcat = pd.Series(os.listdir(path))
            images_per_subcat = shuffle(images_per_subcat, random_state=SHUFFLE_SEED)

            # If there are one or more locations to exclude, exclude them from the list!
            if locations_to_exclude is not None:
                for loc in locations_to_exclude:
                    images_per_subcat = images_per_subcat.loc[~images_per_subcat.str.contains(loc)]

            if samples_to_exclude is not None:
                images_per_subcat = images_per_subcat.loc[~images_per_subcat.isin(samples_to_exclude)]

            # If the dataset is corrected, exclude the corrections
            if len(images_per_subcat) > 0:
                if samples_to_load == 'all':
                    last_img = -1
                else:
                    if self.corrected:
                        # Sort the images, we only want the n first ones
                        order = images_per_subcat.str.split('_', expand=True)[0].astype(int)
                        order = order.sort_values()
                        images_per_subcat = images_per_subcat.reindex(order.index)
                    last_img = samples_to_load_per_subcat[i]

                selected_images = images_per_subcat.iloc[:last_img]
            else:
                selected_images = images_per_subcat
            # If using the corrected dataset, eliminate the ones that are not correct
            if self.corrected and subcat != 'Noise':
                correction_path = os.path.join(self.data_dir, subcat + '2Noise.csv')
                if not os.path.exists(correction_path):
                    raise Exception(
                        'If you want to use the corrected dataset you should provide a csv file with the '
                        'corrections for each of the original classes')
                correction_csv = pd.read_csv(correction_path, header=None)
                all_images_joined_names = selected_images.str.split('_').str.join('')
                selected_images = selected_images.loc[~all_images_joined_names.isin(correction_csv[0])]
            total_selected_subcat += list(selected_images)
        return total_selected_subcat

    def select_data(self, noise_ratio, locations_to_exclude=None):
        """
        The function will return the selected data for all the classes together, shuffled.
        For non-noise classes, the data included will be the first samples_per_class of the dataset (ordered),
        excluding the ones corresponding to the locations to exclude, if any.

        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :param locations_to_exclude: list of locations to not load (for blocked testing)
        :return: x, y and paths
        """
        total_paths = []

        # Loop through all the categories
        for cat_i, category in enumerate(self.int2class):
            # If Noise, select a random amount
            if category == 'Noise':
                samples_to_load = self.get_noise_samples(noise_ratio)
            else:
                samples_to_load = self.samples_per_class
            # Add the data from that category
            print('selecting samples of category %s: %s' % (category, samples_to_load))
            selected_paths = self.select_files_category(category, samples_to_load, locations_to_exclude)
            total_paths += selected_paths

        total_paths = shuffle(total_paths)
        return total_paths

    def read_labels_from_file_list(self, file_list):
        labels = []
        for img_name in file_list:
            category = img_name.split('_')[-1].split('.')[0]
            joined_cat = self.map_join[category]

            # This part is for joined classes
            labels.append(self.classes2int[joined_cat])

        y = np.array(labels)

        return y

    def load_from_file_list(self, file_list):
        labels = []
        images = []
        for img_name in tqdm(file_list, total=len(file_list)):
            category = img_name.split('_')[-1].split('.')[0]
            joined_cat = self.map_join[category]

            img_array = cv2.imread(os.path.join(self.data_dir, category, img_name))

            grey_image = np.mean(img_array, axis=2)
            images.append(grey_image)
            # This part is for joined classes
            labels.append(self.classes2int[joined_cat])

        x = self.reshape_images(images)
        y = np.array(labels)

        return x, y

    def load_set_from_df(self, paths_df, partition):
        print('loading %s set in memory...' % partition)
        paths_list = paths_df.loc[paths_df['set'] == partition, 'path'].values
        return self.load_from_file_list(paths_list)

    def select_more_noise(self, paths_df, new_noise_ratio, partition):
        """
        Append to x_test and y_test more noise, NOT repeated (not the same samples).
        The amount of noise added is according to the new_noise_ratio.

        :param paths_df: pd.DataFrame with all the paths of the images corresponding used
        :param new_noise_ratio: new ratio (0 to 1) of noise from the total dataset
        :param partition: train, valid or test
        :return: updated x_test and y_test
        """
        noise_samples = self.get_noise_samples(new_noise_ratio)
        y_test = self.read_labels_from_file_list(paths_df.loc[paths_df['set'] == partition, 'path'].values)
        new_noise_samples = noise_samples - (y_test == self.classes2int['Noise']).sum()
        selected_paths = self.select_files_category('Noise', samples_to_load=new_noise_samples,
                                                    locations_to_exclude=None,
                                                    samples_to_exclude=paths_df['path'].values)

        new_paths_df = pd.DataFrame({'path': selected_paths})
        new_paths_df['set'] = partition

        paths_df = pd.concat([paths_df, new_paths_df])

        return paths_df

    def get_noise_samples(self, noise_ratio):
        """
        Compute how many noise samples are necessary to get the specified noise_ratio if each class has an amount of
        samples of samples_per_class

        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :return: number of samples
        """
        if noise_ratio == 'all':
            return noise_ratio
        else:
            if self.samples_per_class == 'all':
                samples_per_class = self.how_many_samples()
            else:
                samples_per_class = self.samples_per_class
            return ((len(self.int2class) - 1) * samples_per_class * noise_ratio) / (1 - noise_ratio)

    def batch_load_from_df(self, data_split_df, data_split='test'):
        batch_size = 16

        data_split_test = data_split_df[data_split_df['set'] == data_split]
        images_for_test = pd.Series(data_split_test['path'])

        labels = []
        images = []

        for i, img_path in tqdm(enumerate(images_for_test), total=len(images_for_test)):
            if (i % batch_size == 0) and (i != 0):
                x = self.reshape_images(images)
                y = np.array(labels)
                images = []
                labels = []

                yield x, y, self, images_for_test

            cat_folder = os.path.splitext(os.path.basename(img_path))[0].split('_')[2]
            img_array = cv2.imread(os.path.join(self.data_dir, cat_folder, img_path))

            # Not necessary if images already on the correct format
            # resized_image = cv2.resize(img_array, (self.image_width, self.image_height))
            grey_image = np.mean(img_array, axis=2)
            images.append(grey_image)

            # This part is for joined classes
            labels.append(self.classes2int[self.map_join[cat_folder]])

        x = self.reshape_images(images)
        y = np.array(labels)
        return x, y, self, images_for_test
