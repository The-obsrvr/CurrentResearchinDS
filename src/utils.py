from ray import tune
import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn.metrics import f1_score

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


def evaluate_model(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def lift_analysis(model, x_test, y_test, lgbm, numOfBins=10):
  '''
      add true labels, predicted labels and predicted probablities to the test set.
      we take the abs as we don't care about the positive or negative direction
      but rather the strength/distance. The distance defines how sure the model is of its prediction. 
  '''
  x_set = x_test.copy()
  x_set["true_labels"] = y_test
  x_set["predicted"] = model.predict(x_test)

  if lgbm:
    lgb_predict = model.predict_proba(x_test)
    x_set["predict_prob"] = [max(lgb_predict[i][0] - 0.50, lgb_predict[i][1] - 0.50) \
                             for i in range(len(lgb_predict))]
  else:
    x_set["predict_prob"] = abs(model.decision_function(x_test))

  # sort along decreasing predict probability! 
  x_set.sort_values(by=["predict_prob"], ascending=False, inplace=True)

  # convert the data to bins. We shall be going with the common 10 equi-height approach.
  bin_ht = int(len(x_set)/numOfBins)
  bins = [x_set.iloc[i:i+bin_ht] for i in range(0, len(x_test), bin_ht)]
  # merge the last two bins
  bins[-2] = pd.concat([bins[-2], bins[-1]])
  bins.pop(-1)
  assert len(bins) == numOfBins

  # count the number of correct predictions in each bin
  correct_predictions = []
  for bin in bins:
    seriesObj = bin.apply(lambda x: True 
                          if x['true_labels'] == x['predicted'] 
                          else False, axis=1)
    numOfRows = len(seriesObj[seriesObj == True].index)
    correct_predictions.append(numOfRows)

  # calculate the cumulative gain
  running_correct = np.cumsum(correct_predictions)
  running_correct_percent = np.round((
      running_correct/sum(correct_predictions))*100)

  # prepare bin_summary table
  bins_summ = pd.DataFrame()
  bins_summ["bins"] = range(1, len(bins)+1)
  bins_summ["number_of_cases"] = [len(bins[i]) for i in range(len(bins))]
  bins_summ["correct_predictions"] = correct_predictions
  bins_summ["cumulative_correct"] = running_correct
  bins_summ["cum_correct_percent"] = running_correct_percent

  return bins, bins_summ