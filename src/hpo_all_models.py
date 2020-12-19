from ray import tune
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler
from ray.tune.logger import DEFAULT_LOGGERS
import joblib
import os
import numpy as np
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import lightgbm as lgbm
import math
from sklearn.model_selection import train_test_split
from src.utils import load_data, loguniform_int, sample_array, save_dictionary, save, evaluate_model

EXPERIMENT_NAME = "IML"

x_train_hpo, y_train_hpo = load_data("train")
x_train_hpo, x_val_hpo, y_train_hpo, y_val_hpo = train_test_split(x_train_hpo, y_train_hpo,
                                                                  test_size=0.2, shuffle=False)


def init_model(config: dict = None):
    model_name = config['model_name']

    if model_name == "DecitionTree":
        pass

    elif model_name == "LogisticRegression":
        pass

    elif model_name == "LinearSVR":
        model = svm.LinearSVR(epsilon=config['epsilon'],
                              C=config['C'],
                              loss=config['loss'],
                              max_iter=config['max_iter'],
                              fit_intercept=config['fit_intercept'])

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
        model = MLPRegressor(hidden_layer_sizes=config['hidden_layer_sizes'],
                             alpha=config["alpha"],
                             batch_size=config['batch_size'],
                             learning_rate_init=config["learning_rate_init"],
                             max_iter=config['max_iter'],
                             random_state=config["random_state"]
                             )
    return model


def retrain_best_model(config: dict, split_train_val=True) -> None:
    global x_val
    model_name = config['model_name']

    print("Retrain best", model_name)

    x_train, y_train = load_data("train")
    x_test, y_test = load_data("test")

    if model_name == "MLP":
        config["early_stopping"] = True
        config["validation_fraction"] = 0.2

    if split_train_val:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,
                                                          shuffle=False)

    model = init_model(config)

    df_columns = y_train.columns

    model.fit(x_train, y_train)

    # Logging params
    params = model.get_params()
    save_dictionary(params)

    test_path = "../figures/" + EXPERIMENT_NAME + "/" + model_name + "/test/"
    train_path = "../figures/" + EXPERIMENT_NAME + "/" + model_name + "/train/"
    if split_train_val:
        val_path = "../figures/" + EXPERIMENT_NAME + "/" + model_name + "/validation/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if split_train_val:
        if not os.path.exists(val_path):
            os.makedirs(val_path)

    y_predict_train = model.predict(x_train)
    y_predict_test = model.predict(x_test)
    if split_train_val:
        y_predict_val = model.predict(x_val)

    # Logging metrics
    metrics_test = evaluate_model(y_test.to_numpy(), y_predict_test)
    metrics_train = evaluate_model(y_train.to_numpy(), y_predict_train)
    if split_train_val:
        metrics_val = evaluate_model(y_val.to_numpy(), y_predict_val)

        save(metrics_test)
        save(metrics_train)

        if split_train_val:
            save(metrics_val)

    # Save models to file
    joblib.dump(model,
                "../figures/" + EXPERIMENT_NAME + "/" + model_name + '/model.pkl')

    with open(
            "../figures/" + EXPERIMENT_NAME + "/" + model_name + '/best_hp.csv',
            'w') as f:
        for key in params.keys():
            f.write("%s,%s\n" % (key, params[key]))


def train_model(config: dict) -> None:
    np.random.seed(0)

    # Init model
    model = init_model(config)

    # Train model
    model.fit(x_train_hpo, y_train_hpo)

    # Evaluate model
    acc = evaluate_model(y_val_hpo.to_numpy(), model.predict(x_val_hpo))
    if math.isnan(acc):
        sp_cor = float('-inf')

    tune.report(ACC=acc)


def hpo_model(model_name: str, num_samples: int) -> dict:
    search_space = {}

    if model_name == "LinearSVR":
        search_space = {
            "model_name": model_name,
            "epsilon": tune.uniform(0, 0.5),
            "C": tune.choice(np.arange(1, 100)),
            "loss": tune.choice(["epsilon_insensitive", "squared_epsilon_insensitive"]),
            "max_iter": tune.choice(np.arange(4000, 10000)),
            "fit_intercept": tune.choice([True, False])
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
        #         scheduler=FIFOScheduler(),
        scheduler=ASHAScheduler(metric="ACC",
                                mode="max", ),
        loggers=DEFAULT_LOGGERS,
        resources_per_trial={
            "cpu": 10,
            "gpu": 0
        },
        config=search_space,
        local_dir="/",  # TODO: add directory
    )

    return analysis


def train_baseline():
    for model_name in ["DecitionTree", "LogisticRegression"]:
        print("Train baseline-", model_name)
        try:
            retrain_best_model()
        except Exception as e:
            print("Error", e)


def hpo_all_models() -> None:
    num_samples = 50

    for model_name in ["LGBM", "RandomForest", "AdaBoost", "LinearSVR", "MLP"]:
        print("HPO for ", model_name)
        try:
            analysis = hpo_model(model_name, num_samples)
            retrain_best_model(analysis.get_best_config(metric="ACC"))
        except Exception as e:
            print("Error", e)
