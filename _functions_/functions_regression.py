#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def evaluate_regression(y_real, y_prediction):
    eval_mae = metrics.mean_absolute_error(y_real, y_prediction)
    eval_mse = metrics.mean_squared_error(y_real, y_prediction)
    eval_rmse = np.sqrt(eval_mse)
    eval_r2 = metrics.r2_score(y_real, y_prediction)
    print(f"MAE = {eval_mae}")
    print(f"MSE = {eval_mse}")
    print(f"RMSE = {eval_rmse}")
    print(f"R2 = {eval_r2}")
    return eval_mae, eval_mse, eval_rmse, eval_r2


def evaluate_cross_validation(model, X_cv, y_cv, k):
    cv_scores = cross_val_score(model, X_cv, y_cv,
                            cv=k, scoring='r2')
    print(f"The cross-validation R2 score: {cv_scores.round(2)}")
    cv_res = np.mean(cv_scores).round(3)
    print(f"The average cross-validation R2 score: {cv_res}")
    return cv_res


def evaluate_regression_model(trained_model, X_train, y_train, X_test, y_test):
    print("TRAINING")
    y_pred_train = trained_model.predict(X_train)
    eval_mae_train, eval_mse_train, eval_rmse_train, eval_r2_train = evaluate_regression(y_train, y_pred_train)
    metrics_train = pd.DataFrame([[eval_mae_train, eval_mse_train, eval_rmse_train, eval_r2_train]],
                                 columns = ['MAE', 'MSE', 'RMSE', 'R2'])
    metrics_train['set'] = 'training'
    print(".......")
    print("TEST")
    y_pred_test = trained_model.predict(X_test)
    eval_mae_test, eval_mse_test, eval_rmse_test, eval_r2_test = evaluate_regression(y_test, y_pred_test)
    metrics_test = pd.DataFrame([[eval_mae_test, eval_mse_test, eval_rmse_test, eval_r2_test]],
                                 columns = ['MAE', 'MSE', 'RMSE', 'R2'])
    metrics_test['set'] = 'test'
    return pd.concat([metrics_train, metrics_test])