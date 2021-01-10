import csv
import joblib
import pandas as pd
from utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sklearn
import shap

shap.initjs()
x_train, x_test, y_train, y_test  = load_data()

with open('../results/retrain_best_svm_with_prob/SVR/model.pkl', 'rb') as file:
    svr = joblib.load(file)
    
x_test_bar = shap.sample(x_test, 1000)
svm_explainer = shap.KernelExplainer(svr.predict_proba, x_test)
svm_shap_values = svm_explainer.shap_values(x_test_bar, nsamples=20)

try:
    import cPickle as pickle
except BaseException:
    import pickle


file_path0 = '../results/Methods/SVR/svm_shap_values.pkl'
with open(file_path0, "wb") as f:
    pickle.dump(svm_shap_values, f)

file_path1 = '../results/Methods/SVR/x_test_bar.pkl'
with open(file_path1, "wb") as f:
    pickle.dump(x_test_bar, f)