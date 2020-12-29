from ray import tune
import numpy as np
import pandas as pd
from numpy.random import randint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def load_data(path:str = '../data/bank_one_hot.csv'):
    """
    :type split: str ["train", "test"]
    Return:
        trainset, testset
    """
    bank_model_data = pd.read_csv(path, sep=",", index_col=0)
    X = bank_model_data.drop(columns=["y_yes"])
    y = bank_model_data["y_yes"]
    num_features = ["age", "campaign", "previous", "emp.var.rate", "cons.price.idx", 
                    "cons.conf.idx", "euribor3m", "nr.employed"]
    scaler = StandardScaler(with_mean=True,with_std=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify= y)
    scaler.fit(x_train[num_features])
    x_train[num_features] = scaler.transform(x_train[num_features])
    x_test[num_features] = scaler.transform(x_test[num_features])
    
    return x_train, x_test, y_train, y_test 

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

def evaluate_model(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def save(metrics_test):
    pass
