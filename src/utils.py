from ray import tune
import numpy as np
from numpy.random import randint

def load_data(split: str = "Train"):
    """
    :type split: str ["train", "test"]
    """
    pass

"""
Provides functionality to fastly include big arrays for choice into HPO with ray.tune.
"""
def loguniform_int(min_bound: int, max_bound: int) -> tune.sample_from:   # noqa
    """Sugar for sampling in different orders of magnitude.

    Args:
        min_bound (float): Lower boundary of the output interval (1e-4)
        max_bound (float): Upper boundary of the output interval (1e-2)
        base (float): Base of the log. Defaults to 10.
    Return:
        int
    """
    base = 10
    logmin = np.log(min_bound) / np.log(base)
    logmax = np.log(max_bound) / np.log(base)

    def apply_log(_):
        return int(base**(np.random.uniform(logmin, logmax)))

    return tune.sample_from(apply_log)


def sample_array(lower_bound: int, upper_bound: int, length: int,
                 ascending: bool = True):
    """ Sample function for sampling arrays in tune.sample_from()
    """
    # pylint: disable=no-else-return
    if ascending:
        return np.sort(32 * randint(lower_bound, upper_bound, size=(length,)))
    else:
        return np.sort(32 * randint(lower_bound, upper_bound, size=(length,)))[::-1]  # noqa: E501

def save_dictionary(params):
    pass

def evaluate_model(param, y_predict_test):
    pass

def save(metrics_test):
    pass