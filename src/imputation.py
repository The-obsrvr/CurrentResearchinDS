import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from fancyimpute import KNN, IterativeImputer


bank_data = pd.read_csv("../data/bank-additional-full.csv", sep=";")

# create ordinal encoder
coded_bank_data = bank_data.copy()
ordinal_enc_dict = {}

col_names = ["education", "job", "marital", "default", "loan", "housing",
             "contact", "month", "day_of_week", "poutcome", "y"]

for col_name in col_names:

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
        coded_bank_data[col_name][value] = encoded_vals[i]

coded_bank_data = coded_bank_data.replace({"unknown": np.nan})

KNN_Imputer = IterativeImputer()

bank_data_imp = coded_bank_data.copy(deep=True)

bank_data_imp.iloc[:,:] = np.round(KNN_Imputer.fit_transform(bank_data_imp))

for col in col_names:

  reshaped_col = bank_data_imp[col].values.reshape(-1, 1)
  bank_data_imp[col] = ordinal_enc_dict[col].inverse_transform(reshaped_col)


bank_data_imp.to_csv("../data/imputed_bank_data_mice.csv")