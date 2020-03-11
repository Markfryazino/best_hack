import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def rescaled_mae(y_pred, y_true, scaler):
    y_true_rescaled = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
    return mean_absolute_error(y_true_rescaled, y_pred_rescaled)


class Model:
    def get_first_level_matrix(self, X):
        preds = {}
        for model, name in self.models:
            preds[name] = model.predict(X)
        return pd.DataFrame(preds).values

    def __init__(self, scaler, xgb_params={}, knn_params={}, nn_params={},
                 svm_params={}, forest_params={}, meta_params={}):
        if forest_params is None:
            forest_params = {}
        self.xgb = XGBRegressor(**xgb_params)
        self.knn = KNeighborsRegressor(**knn_params)
        self.svm = SVC(**svm_params)
        self.nn = MLPRegressor(**nn_params)
        self.forest = RandomForestRegressor(**forest_params)
        self.meta = XGBRegressor(**meta_params)
        self.scaler = scaler
        self.models = [(self.xgb, 'xgboost'), (self.knn, 'knn'), (self.svm, 'svm'),
                       (self.nn, 'neural_network'), (self.forest, 'random_forest')]

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        for model, name in self.models:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            print('MAE of ' + name + ' on train: ', rescaled_mae(y_train_pred, y_train, self.scaler))
            if X_test is not None:
                y_test_pred = model.predict(X_test)
                print('MAE of ' + name + ' on test: ', rescaled_mae(y_test_pred, y_test,
                                                                    self.scaler), '\n')

        first_level = self.get_first_level_matrix(X_train)
        self.meta.fit(first_level, y_train)
        y_train_pred = self.meta.predict(first_level)
        print('MAE of meta model on train: ', rescaled_mae(y_train_pred, y_train, self.scaler))
        if X_test is not None:
            y_test_pred = self.meta.predict(self.get_first_level_matrix(X_test))
            print('MAE of meta model on test: ', rescaled_mae(y_test_pred, y_test, self.scaler), '\n')


