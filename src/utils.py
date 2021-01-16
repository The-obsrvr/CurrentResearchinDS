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


class DataGenerator():

  def __init__(self, datapath, imputed, preprocessed):
    super(DataGenerator, self).__init__()

    self.imputed = imputed
    self.preprocessed = preprocessed
    if not self.preprocessed and not self.imputed:
      self.bank_data = pd.read_csv(datapath, sep=";")
    else:
      self.bank_data = pd.read_csv(datapath, index_col=0)
    self.cat_features = ["education", "job", "marital",  "month", "day_of_week",
                         "binned_duration", "poutcome", "housing", "loan", "contact"]
    self.num_features = ["age", "campaign", "previous", "emp.var.rate", "cons.price.idx", 
                    "cons.conf.idx", "euribor3m", "nr.employed"]
    self.target = ["y"]
    
    self.scaler = StandardScaler(with_mean=True,with_std=True)
    self.scaler.fit(self.bank_data[self.num_features])


  def impute_function(self):

    # Define the ordinal encoder
    encoded_bank_data = self.bank_data.copy()
    ordinal_enc_dict = {}

    # use ordinal encoder to encode all the categorical columns
    for col_name in self.cat_features:

      ordinal_enc_dict[col_name] = OrdinalEncoder()

      col = bank_data[col_name]
      list_of_indices_not_na = []
      for i, value in enumerate(col):
        if value != "unknown":
          list_of_indices_not_na.append(i)

      col.replace({"unknown": np.nan}, inplace=True)
      col_notna = col[col.notna()]
      reshaped_colvalues = col_notna.values.reshape(-1, 1)

      encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_colvalues)

      for i, value in enumerate(list_of_indices_not_na):
        encoded_bank_data[col_name][value] = encoded_vals[i]

    encoded_bank_data = encoded_bank_data.replace({"unknown": np.nan})
    # define the Imputer Object
    MICE_Imputer = IterativeImputer()

    imputed_bank_data = encoded_bank_data.copy(deep=True)

    # fit the imputation values
    imputed_bank_data.iloc[:, :] = np.round(MICE_Imputer.fit_transform(imputed_bank_data))

    # reverse the ordinal encoding
    for col in self.cat_features:
      reshaped_col = imputed_bank_data[col].values.reshape(-1, 1)
      imputed_bank_data[col] = ordinal_enc_dict[col].inverse_transform(reshaped_col)
    
    return imputed_bank_data


  def outlier_removal(self, column, quantile = 0.99):

    q = self.bank_data[column].quantile(quantile)
    return  self.bank_data[self.bank_data[column] < q]


  def bin_duration(self):

    duration_bin_labels = ['less_than_200sec', '200sec_400sec', '400sec_800sec', 
                    'more_than_800sec']
    dur_bin_size = [-1, 200, 400, 800, 200000]

    # remove outliers
    self.bank_data = self.outlier_removal(column="duration", quantile=0.99)

    # bin duration
    self.bank_data["binned_duration"] = pd.cut(self.bank_data["duration"],
                                               bins=dur_bin_size,
                                               labels=duration_bin_labels)
    # drop the original duration column
    self.bank_data.drop(columns=["duration"], inplace=True)


  def define_cat_maps(self):

    self.category_map = {}
    for i, feature in enumerate(self.cat_features + self.target):
      if feature == "binned_duration":
        self.category_map[i] = ["less_than_200sec", "200_400sec", "400-800sec",
                                "more_than_800sec"]
        continue
      self.category_map[i] = sorted(self.bank_data[feature].unique())

    self.cat_vars_ord ={}
    for i in range(len(list(self.category_map.keys())) - 1):
      self.cat_vars_ord[i] = len(self.category_map[i]) 

    self.cat_vars_ohe = {}
    for i, (_,v) in enumerate(self.cat_vars_ord.items()):
      if i == 0:
        prev_key = 0 
        prev_value = v
        self.cat_vars_ohe[i] = v
        continue
      self.cat_vars_ohe[prev_key + prev_value] = v
      prev_key = prev_key + prev_value 
      prev_value = v


  def data_preprocessing(self):
    '''
    '''
    
    # Step 1: Drop Duplicates
    self.bank_data.drop_duplicates(subset=None, keep='first', inplace=True)

    # Step 2: Drop non-relevant columns as decided by initial EDA
    self.bank_data.drop(columns=["default", "pdays"], inplace = True)

    # Step 3: Imputation
    if not self.imputed:
      self.bank_data = self.impute_function()

    # Step 4: Bin Duration and drop Outliers
    self.bin_duration()

    # Step 5: Edit some Categorical Features values
    self.bank_data.loc[self.bank_data['education'] == 'basic.9y','education'] = 'basic'
    self.bank_data.loc[self.bank_data['education'] == 'basic.6y','education'] = 'basic'
    self.bank_data.loc[self.bank_data['education'] == 'basic.4y','education'] = 'basic'
    self.bank_data['education'] = self.bank_data['education'].str.replace('.','')
    self.bank_data['job'] = self.bank_data['job'].str.replace('.','')

    # Step 6: One-Hot Encode Categorical Features
    
    # self.cat_vars_ohe = ord_to_ohe(np.array(self.bank_data), self.cat_vars_ord)[1]

    self.define_cat_maps()

    self.bank_data = pd.get_dummies(self.bank_data, columns=self.cat_features)
    self.bank_data = pd.get_dummies(self.bank_data, columns=self.target, drop_first=True)
       

  def load_data(self):

    if not self.preprocessed:
      self.data_preprocessing()
    
    X = self.bank_data.drop(columns=["y_yes"])
    y = self.bank_data["y_yes"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)
    
    x_train[self.num_features] = self.scaler.transform(x_train[self.num_features])
    x_test[self.num_features] = self.scaler.transform(x_test[self.num_features])
    
    # reorder to place one-hot encoded features at the start
    x_train = x_train[[c for c in x_train if c not in self.num_features]
                      + self.num_features]
    x_test = x_test[[c for c in x_train if c not in self.num_features]
                      + self.num_features]
    
    return x_train, x_test, y_train, y_test


  def reverse_cat_features(data):

    return data


  def rev_transform(self, data, rev_num_features=True, rev_cat_features=True):

    if rev_num_features:
      data[self.num_features] = self.scaler.inverse_transform(data[self.num_features])
    if rev_cat_features:
      data[self.cat_features] = self.reverse_cat_features(data[self.cat_features])
    
    return data