from ray import tune
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler
from ray.tune.logger import DEFAULT_LOGGERS
import joblib
import os
import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
import lightgbm as lgbm
import math
from sklearn.model_selection import train_test_split
from utils import load_data, loguniform_int, sample_array, evaluate_model
from data_generator import DataGenerator
import csv

result_dir = '../results/'
dataGeneratorinit = DataGenerator('../data/imputed_bank_data_mice.csv', imputed=True, preprocessed=False)
x_train, x_test, y_train, y_test  = dataGeneratorinit.load_data()
x_train_hpo, x_val_hpo, y_train_hpo, y_val_hpo = train_test_split(x_train, y_train,test_size=0.2, shuffle=False)

def init_model(config: dict = None):
    '''
        Init Model based on configuration.
    '''
    model_name = config['model_name']

    if model_name == "DecisionTree":
        model = tree.DecisionTreeClassifier()

    elif model_name == "LogisticRegression":
        model = LogisticRegression(random_state=42, max_iter=1000)

    elif model_name == "SVR":
        model = svm.SVC(kernel=config['kernel'],
                              C=config['C'],
                              degree=config['degree'],
                              max_iter=config['max_iter'],
                              probability=config['probability'])

    elif model_name == "LGBM":
        model = lgbm.LGBMClassifier(n_jobs=config['n_jobs'],
                                    num_leaves=config['num_leaves'],
                                    learning_rate=config['learning_rate'],
                                    n_estimators=config['n_estimators'],
                                    subsample_for_bin=config['subsample_for_bin'],
                                    )
    elif model_name == "RandomForest":
        model = RandomForestClassifier(n_jobs=-1,
                                       max_leaf_nodes=config['max_leaf_nodes'],
                                       min_impurity_decrease=config['min_impurity_decrease'],
                                       max_samples=config['max_samples'],
                                       n_estimators=config['n_estimators'],
                                       min_samples_split=config['min_samples_split'],
                                       min_samples_leaf=config['min_samples_leaf']
                                       )
    elif model_name == "AdaBoost":
        model = AdaBoostClassifier(n_estimators=config['n_estimators'],
                                   learning_rate=config['learning_rate'],
                                   )

    elif model_name == "MLP":
        model = MLPClassifier(hidden_layer_sizes=config['hidden_layer_sizes'],
                             alpha=config["alpha"],
                             batch_size=config['batch_size'],
                             learning_rate_init=config["learning_rate_init"],
                             max_iter=config['max_iter'],
                             random_state=config["random_state"]
                             )
    return model


def retrain_best_model(config: dict, exp) -> None:
    '''
        Retrain the best configuration got from HPO and save the model.
    '''
    model_name = config['model_name']
    print("Retrain best", model_name)

    if model_name == "MLP":
        config["early_stopping"] = True
        config["validation_fraction"] = 0.2
        config['max_iter'] = 1000

    model = init_model(config)
    model.fit(x_train, y_train)

    # Logging params
    params = model.get_params()

    y_predict_test = model.predict(x_test)
    macro_f1_test = evaluate_model(y_test.to_numpy(), y_predict_test)
    
    model_path = result_dir + exp + "/" + model_name + "/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Save models to file
    joblib.dump(model, model_path+'model.pkl')

    with open(result_dir + model_path + 'best_hp.csv','w') as f:
        for key in params.keys():
            f.write("%s,%s\n" % (key, params[key]))
            
    return macro_f1_test


def retrain_best_model_2(config: dict, exp) -> None:
    '''
        Retrain the best configuration got from HPO and save the model.
    '''
    model_name = config['model_name']
    print("Retrain best", model_name)
    
    dataGenerator = DataGenerator('../data/imputed_bank_data_mice.csv', True, False)
    x_train_2, x_test_2, y_train_2, y_test_2  = dataGenerator.load_data()

    if model_name == "MLP":
        config["early_stopping"] = True
        config["validation_fraction"] = 0.2
        config['max_iter'] = 1000

    model = init_model(config)
    model.fit(x_train_2, y_train_2)

    # Logging params
    params = model.get_params()

    y_predict_test = model.predict(x_test_2)
    macro_f1_test = evaluate_model(y_test_2.to_numpy(), y_predict_test)
    
    model_path = result_dir + exp + "/" + model_name + "/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Save models to file
    joblib.dump(model, model_path+'model.pkl')

    with open(result_dir + model_path + 'best_hp.csv','w') as f:
        for key in params.keys():
            f.write("%s,%s\n" % (key, params[key]))
            
    return macro_f1_test


def train_model(config: dict) -> None:
    '''
        Train model based on config
    '''
    np.random.seed(0)

    # Init model
    model = init_model(config)

    # Train model
    model.fit(x_train_hpo, y_train_hpo)

    # Evaluate model
    macro_f1_test = evaluate_model(y_val_hpo.to_numpy(), model.predict(x_val_hpo))

    tune.report(macro_f1_test=macro_f1_test)


def hpo_model(model_name: str, num_samples: int) -> dict:
    '''
        Hyper parameter optimization.
    '''
    search_space = {}

    if model_name == "SVR":
        search_space = {
            "model_name": model_name,
            "kernel": tune.choice(['linear', 'poly', 'rbf', 'sigmoid']),
            "C": tune.choice(np.arange(1, 100)),
            "max_iter": tune.choice(np.arange(4000, 10000)),
            "degree": tune.choice(np.arange(2, 7)),
            'probability': True,
        }

    elif model_name == "LGBM":
        search_space = {
            "model_name": model_name,
            "n_jobs": -1,
            "num_leaves": tune.choice(np.arange(15, 100)),
            "learning_rate": tune.uniform(0.00001, 0.1),
            "n_estimators": tune.choice(np.arange(50, 500)),
            "subsample_for_bin": tune.choice(np.arange(50000, 400000))
        }

    elif model_name == "RandomForest":
        search_space = {
            "model_name": model_name,
            "n_jobs": -1,
            "max_leaf_nodes": loguniform_int(10, 500),
            "min_impurity_decrease": tune.uniform(0.0, 0.1),
            "max_samples": tune.uniform(0.05, 0.3),
            "n_estimators": loguniform_int(50, 500),
            "min_samples_split": tune.uniform(0, 1),
            "min_samples_leaf": tune.uniform(0, 0.4)
        }

    elif model_name == "AdaBoost":
        search_space = {
            "model_name": model_name,
            "n_jobs": -1,
            "n_estimators": loguniform_int(50, 500),
            "learning_rate": tune.uniform(0.3, 1),
        }

    elif model_name == "MLP":
        search_space = {
            "model_name": model_name,
            "number_of_layers": tune.sample_from([2, 5]),
            "hidden_layer_sizes": (tune.sample_from(
                lambda spec: sample_array(1, 8, spec.config.number_of_layers, False))),
            # noqa: E501
            "alpha": tune.uniform(0, 0.01),
            "batch_size": loguniform_int(32, 256),
            "learning_rate_init": tune.uniform(0.000001, 0.001),
            "max_iter": 20,
            "random_state": 42

        }

    analysis = tune.run(
        train_model,
        num_samples=num_samples,
        verbose=2,
        scheduler=ASHAScheduler(metric="macro_f1_test",
                                mode="max", ),
        loggers=DEFAULT_LOGGERS,
        resources_per_trial={
            "cpu": 5,
            "gpu": 0
        },
        config=search_space,
        local_dir=result_dir,
    )

    return analysis


def train_baseline():
    '''
        Train base line models.
    '''
    for model_name in ["DecisionTree", "LogisticRegression"]:
        print("Train baseline-", model_name)
        config = {}
        config['model_name'] = model_name
        try:
            retrain_best_model(config, "Models")
        except Exception as e:
            print("Error", e)


def hpo_all_models() -> None:
    '''
        Perform HPO for all models.
        For each model it will try 200 different combination of Hyper-parameters.
    '''
    with open(result_dir+'test_performance.csv', 'a+', newline='') as file:
        writer = csv.writer(file)  
        for model_name in [ "MLP", "SVR", "LGBM", "RandomForest", "AdaBoost"]:
            print("HPO for ", model_name)
            try:
                num_samples = 100 if model_name == "MLP" else 200
                analysis = hpo_model(model_name, num_samples)
                macro_f1_test = retrain_best_model(analysis.get_best_config(metric="macro_f1_test"), "Models")
                writer.writerow([model_name, macro_f1_test])
            except Exception as e:
                print("Error", e)

def retrain_best_lgbm_with_binary_cat() -> None:
    best_hp = { 'model_name' : 'LGBM',
                'boosting_type': 'gbdt',
                 'class_weight': None,
                 'colsample_bytree': 1.0,
                 'importance_type': 'split',
                 'learning_rate': 0.08978117806420283,
                 'max_depth': -1,
                 'min_child_samples': 20,
                 'min_child_weight': 0.001,
                 'min_split_gain': 0.0,
                 'n_estimators': 76,
                 'n_jobs': -1,
                 'num_leaves': 57,
                 'objective': None,
                 'random_state': None,
                 'reg_alpha': 0.0,
                 'reg_lambda': 0.0,
                 'silent': True,
                 'subsample': 1.0,
                 'subsample_for_bin': 272866,
                 'subsample_freq': 0}
    macro_f1_test = retrain_best_model_2(best_hp, 'retrain_best_lgbm_with_binary_cat')
    print(f'macro_f1_test:{macro_f1_test}')
                
        
if __name__ == "__main__":
    train_baseline()
    hpo_all_models()
    retrain_best_lgbm_with_binary_cat()
