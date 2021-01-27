import numpy as np
import pandas as pd
from fancyimpute import IterativeImputer
from alibi.utils.mapping import ohe_to_ord
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler 


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
    cat_features = ["education", "job", "marital", "loan", "housing",
             "contact", "month", "day_of_week", "poutcome", "y"]
    # use ordinal encoder to encode all the categorical columns
    for col_name in cat_features:

      ordinal_enc_dict[col_name] = OrdinalEncoder()

      col = self.bank_data[col_name]
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
    for col in cat_features:
      reshaped_col = imputed_bank_data[col].values.reshape(-1, 1)
      imputed_bank_data[col] = ordinal_enc_dict[col].inverse_transform(reshaped_col)

    # save a copy of the imputed data for future use. 
    imputed_bank_data.to_csv("bank_data_mice.csv", sep=";", index=False)
    
    return imputed_bank_data


  def outlier_removal(self, column, quantile = 0.99):

    q = self.bank_data[column].quantile(quantile)
    return  self.bank_data[self.bank_data[column] < q]


  def bin_duration(self):

    duration_bin_labels = ['less than 200sec', '200sec - 400sec', '400sec - 800sec', 
                    'more than 800sec']
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
        self.category_map[i] = ['200sec - 400sec', '400sec - 800sec',
                                'less than 200sec', 'more than 800sec']
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
    self.bank_data['education'] = self.bank_data['education'].str.replace('.',' ')
    self.bank_data['job'] = self.bank_data['job'].str.replace('.',' ')

    # Step 6: One-Hot Encode Categorical Features
    
    self.define_cat_maps()

    self.pre_one_hot_encoded_data = self.bank_data.copy()
    self.pre_one_hot_encoded_data.drop(columns=["y"], inplace=True)
    self.pre_one_hot_encoded_data = self.pre_one_hot_encoded_data[[c 
                                                                   for c in self.pre_one_hot_encoded_data
                                                                   if c not in self.num_features] 
                                                                  + self.num_features]
    self.pre_one_hot_encoded_data.reset_index(drop=True, inplace=True)

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
    
    self.column_names = x_train.columns

    return x_train, x_test, y_train, y_test


  def reverse_cat_features(self, data):

    data = np.array(data).reshape((1, len(self.column_names)))
    X_orig_ord = ohe_to_ord(data, self.cat_vars_ohe)[0]
    cat_vars = []
    for i, (_, v) in enumerate(self.category_map.items()):

      cat_orig = v[int(X_orig_ord[0, i])]
      cat_vars.append(cat_orig)

    return cat_vars[:-1]


  def rev_transform(self, data, rev_num_features=True, rev_cat_features=True):
    '''
    accepts one dataframe row containing one hot encoded values. Shape (1, 55)
    '''

    if rev_cat_features and rev_num_features:
      feature_values = self.reverse_cat_features(data) + list(self.scaler.inverse_transform(data[self.num_features]))
      data = pd.DataFrame([feature_values], columns=self.cat_features + self.num_features)

    elif rev_num_features:
      data[self.num_features] = self.scaler.inverse_transform(data[self.num_features])
    
    else:
      feature_values = self.rev_cat_features(data) + list(data[self.num_features])
      data = pd.DataFrame([feature_values], columns=self.cat_features + self.num_features)
    return data
  

  def transform(self, instance):

    self.pre_one_hot_encoded_data.loc[len(self.pre_one_hot_encoded_data)] = instance
    encoded_data = pd.get_dummies(
        self.pre_one_hot_encoded_data, columns=self.cat_features)
    encoded_data[self.num_features] = self.scaler.transform(encoded_data[self.num_features])
    encoded_data = encoded_data[[c for c in encoded_data 
                                 if c not in self.num_features] 
                                + self.num_features]

    return encoded_data.loc[len(encoded_data)-1]