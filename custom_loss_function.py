
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from typing import Tuple

# If prediction matches reality... it is good news so assign a standard weight of 1
# If call types are confused with each other we want e medium penalty
# If noise is predicted as call types we want a high penalty
# If call types are predicted as noise we want a low penalty
scenario_risk = {'prediction_matches_reality': 1.0,
                 'call_type_confusion': 3.0,
                 'noise_as_call_confusion': 15.0,
                 'calls_as_noise_confusion': 1.5}

scenario_risk_normalised = {}
worst_case_scenario_risk = np.array(list(scenario_risk.values())).max()

for key, value in scenario_risk.items():
    scenario_risk_normalised[key] = value / worst_case_scenario_risk


@tf.function
def _convert_scenario_risk_to_tensor(scenario: str) -> tf.Tensor:
    """
    Look up how risky the scenario is when comparing the model prediction to reality, and return a weight score
    indicating how much emphasis the model should place on learning from that particular observation.

    Parameters
    ----------
    scenario : str
        Key of dictionary which shows the risk attached to each scenario. Should be one of...
            'prediction_matches_reality'
            'less_spend_than_expected'
            'much_less_spend_than_expected'
            'more_spend_than_expected'
            'much_more_spend_than_expected'

    Returns
    -------
    tf.Tensor
        Weight which signifies how bad the scenario is for the business, and thus how to update the loss which the
        model is trying to improve.
    """

    return tf.cast(x=tf.constant(scenario_risk_normalised[scenario]), dtype=tf.float32)


@tf.function
def _get_loss_adjustment_for_scenario(actual_vs_predicted_class: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """
    Return a weight according to how damaging the scenario is to the business.

    Parameters
    ----------
    actual_vs_predicted_class : Tuple[tf.Tensor, tf.Tensor]
       The actual class, actual_vs_predicted_class[0], and the predicted class, actual_vs_predicted_class[1], for an
       observation.

    Returns
    -------
    float
        Weight which signifies how bad the scenario is for the business, and thus how to upweight the loss
        which the model should be trying to improve.
    """

    actual_class = actual_vs_predicted_class[0]
    predicted_class = actual_vs_predicted_class[1]

    # Retrieve the appropriate weighting based on how the prediction compares with reality

    return tf.case(
        [
            (tf.equal(actual_class, predicted_class),
             lambda: _convert_scenario_risk_to_tensor('prediction_matches_reality')),

            (tf.logical_and(tf.logical_and(tf.math.not_equal(actual_class, 3), tf.math.not_equal(predicted_class, 3)), tf.math.not_equal(predicted_class, actual_class)),
             lambda: _convert_scenario_risk_to_tensor('call_type_confusion')),

            (tf.logical_and(tf.math.not_equal(actual_class, 3), tf.equal(predicted_class, 3)),
             lambda: _convert_scenario_risk_to_tensor('calls_as_noise_confusion')),

            (tf.logical_and(tf.equal(actual_class, 3), tf.math.not_equal(predicted_class, 3)),
             lambda: _convert_scenario_risk_to_tensor('noise_as_call_confusion')),

        ]
    )


@tf.function
def custom_cross_entropy(y_actual: tf.Tensor, y_prediction: tf.Tensor) -> tf.Tensor:
    """
    Calculate cross entropy loss, but weighted according to how risky the scenario is.

    Parameters
    ----------
    y_actual : tf.Tensor
        One-hot encoded representation of the true class for each observation.
    y_prediction : tf.Tensor
        The predicted probability that the observation belongs to each class.

    Returns
    -------
    tf.Tensor
        Weighted version of cross-entropy loss depending on how the prediction compares to reality.
    """

    standard_cross_entropy = losses.sparse_categorical_crossentropy(y_true=y_actual, y_pred=y_prediction)

    # The label is one hot encoded, so identify the true class by which element contains the maximum value (1)
    actual_class = tf.math.argmax(input=y_actual, axis=1)
    actual_class = tf.cast(x=actual_class, dtype=tf.float32)

    # The prediction is a distribution of probabilities per class,
    # so identify the predicted class by which element contains the maximum probability
    predicted_class = tf.math.argmax(input=y_prediction, axis=1)
    predicted_class = tf.cast(x=predicted_class, dtype=tf.float32)

    # actual_vs_predicted_class = tf.stack(values=[actual_class, predicted_class], axis=1)

    # Calculate the weighting that should be applied to each observation in the data
    # weighting = tf.map_fn(fn=_get_loss_adjustment_for_scenario, elems=actual_vs_predicted_class)
    weighting = number_comparison(actual_class, predicted_class)

    # Updated the cross entropy loss based on the weighting
    return tf.math.multiply(standard_cross_entropy, weighting)


@tf.function
def number_comparison(actual, predicted):
    new_tensor = tf.math.subtract(tf.math.exp(actual), tf.math.exp(predicted))
    weightings = tf.zeros([tf.size(new_tensor)])
    weightings = tf.where(new_tensor == 0, 0.07, weightings)
    weightings = tf.where(tf.less(new_tensor, 7) & tf.math.not_equal(new_tensor, 0) & tf.greater(new_tensor,-10), 0.2, weightings)
    weightings = tf.where(new_tensor < -10, 0.1, weightings)
    weightings = tf.where(new_tensor > 10, 1., weightings)

    return weightings
