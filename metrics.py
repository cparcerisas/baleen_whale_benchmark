import numpy as np
import tensorflow as tf
import keras.backend as K
from sklearn.metrics import confusion_matrix


class ImbalancedDetectionMatrix(object):
    def __init__(self, noise_class_name, classes_names):
        self.classes_names = classes_names
        self.not_noise_indices = []
        for class_i, name in enumerate(classes_names):
            if name != noise_class_name:
                self.not_noise_indices.append([class_i])
            else:
                self.noise_class_index = [class_i]

    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred = tf.argmax(y_pred, 1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=len(self.classes_names))
        return cm

    def imbalanced_metric(self, y_true, y_pred):
        cm = self.confusion_matrix(y_true, y_pred)
        cm_calls = tf.gather_nd(cm, indices=self.not_noise_indices)
        cm_noise_total = tf.gather_nd(cm, indices=self.noise_class_index)
        cm_noise = tf.gather_nd(cm_noise_total, indices=self.not_noise_indices)
        diag_part = tf.linalg.diag_part(cm_calls)
        detections = tf.reduce_sum(cm_calls, 1) + tf.constant(1e-15)
        call_avg_tpr = tf.reduce_mean(diag_part / detections)
        noise_misclassification_rate = 1 - tf.reduce_sum(cm_noise) / (tf.reduce_sum(cm_noise_total, 0) +
                                                                      tf.constant(1e-15))
        imbalanced_metric = 2 * (call_avg_tpr * (noise_misclassification_rate**2)) / (call_avg_tpr +
                                                                                 noise_misclassification_rate)
        return imbalanced_metric

    def call_avg_tpr(self, y_true, y_pred):
        # cm = self.total_cm
        cm = self.confusion_matrix(y_true, y_pred)

        cm_calls = tf.gather_nd(cm, indices=self.not_noise_indices)
        diag_part = tf.linalg.diag_part(cm_calls)
        detections = tf.reduce_sum(cm_calls, 1) + tf.constant(1e-15)
        call_avg_tpr = tf.reduce_mean(diag_part / detections)
        return call_avg_tpr

    def noise_misclas_rate(self, y_true, y_pred):
        # cm = self.total_cm
        cm = self.confusion_matrix(y_true, y_pred)

        cm_noise_total = tf.gather_nd(cm, indices=self.noise_class_index)
        cm_noise = tf.gather_nd(cm_noise_total, indices=self.not_noise_indices)
        noise_misclassification_rate = tf.reduce_sum(cm_noise) / (tf.reduce_sum(cm_noise_total, 0) + tf.constant(1e-15))
        return noise_misclassification_rate


def recall_score(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall_val = true_positives / (all_positives + K.epsilon())
    return recall_val


def precision_score(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_val = true_positives / (predicted_positives + K.epsilon())
    return precision_val


def f1_score(y_true, y_pred):
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))
