import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from .semifinal_features import prepare_features, prepare_reduced_features
from sklearn.metrics import mean_absolute_error
from .semifinal_model import Model
from sklearn.model_selection import train_test_split


def validate_model(raw, model_params, use_subset=False):
    if use_subset:
        print('ВАЛИДАЦИЯ ДОПОЛНИТЕЛЬНОЙ ЗАДАЧИ №2')
    else:
        print('ВАЛИДАЦИЯ ОСНОВНОЙ ЗАДАЧИ')
    kf = KFold(n_splits=5, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kf.split(raw)):
        print('ITERATION ' + str(i))
        train = raw.loc[train_idx].copy().reset_index(drop=True)
        test = raw.loc[test_idx].copy().reset_index(drop=True)
        y_test = test['Energ_Kcal'].values
        test.drop('Energ_Kcal', axis=1, inplace=True)

        if not use_subset:
            train_proc, test_proc, scaler = prepare_features(train, test, False)
        else:
            train_proc, test_proc, scaler = prepare_reduced_features(train, test, False)
        model_params['scaler'] = scaler
        X_train = train_proc.drop('Energ_Kcal', axis=1).values
        y_train = train_proc['Energ_Kcal'].values
        X_tt, X_tv, y_tt, y_tv = train_test_split(X_train, y_train, test_size=0.2)
        X_test = test_proc.values

        model = Model(**model_params)
        model.fit(X_tt, y_tt, X_tv, y_tv)
        y_pred = model.predict_ready(X_test)
        print('Mean absolute error on validation: ', mean_absolute_error(y_test, y_pred), '\n\n')


def end_to_model(train, test, model_params, use_subset=False):
    if not use_subset:
        train_proc, test_proc, scaler = prepare_features(train, test)
    else:
        train_proc, test_proc, scaler = prepare_reduced_features(train, test)
    model_params['scaler'] = scaler
    X_train = train_proc.drop('Energ_Kcal', axis=1).values
    y_train = train_proc['Energ_Kcal'].values
    X_test = test_proc.values

    model = Model(**model_params)
    model.fit(X_train, y_train)
    test_pred = model.predict_ready(X_test)
    answer = pd.Series(test_pred.reshape(-1), name='Pred_kcal')
    return model, answer
