# Package Imports
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, root_mean_squared_error
import xgboost as xgb
import shap
from imblearn.under_sampling import RandomUnderSampler


def create_Telco_xgb_train_test(data: pd.DataFrame, random_under_sample: bool = True, columns_to_drop: list = None,
                                test_size: float = 0.1, random_state: int = 42):
    """Creates the train and test sets for the Telco dataset to be used in xgboost. If columns_to_drop are given, these
    columns are dropped from the dataset."""

    # Splitting the data into train and test sets
    X = data.drop(['customerID', 'Churn'], axis=1)  # Features
    y = data['Churn']  # Target variable

    # Random under sampling (the training set) to balance the dataset
    if random_under_sample:
        rus = RandomUnderSampler(random_state=random_state, replacement=False)
        X, y = rus.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Perform one-hot encoding for categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_features)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_features)

    if columns_to_drop is not None:
        # Drop non-important features from X_train and X_test
        X_train_encoded = X_train_encoded.drop(columns=columns_to_drop)
        X_test_encoded = X_test_encoded.drop(columns=columns_to_drop)

    return X_train_encoded, X_test_encoded, y_train, y_test


def create_xgb_train_test(data: pd.DataFrame, drop_x: list[str], target_name: str, random_under_sample: bool = True,
                          shap_columns_to_drop: list = None, test_size: float = 0.1, random_state: int = 42):
    """Creates the train and test sets for the Telco dataset to be used in xgboost. If columns_to_drop are given, these
    columns are dropped from the dataset."""

    # Splitting the data into train and test sets
    X = data.drop(drop_x, axis=1)  # Features
    y = data[target_name]  # Target variable

    # Random under sampling (the training set) to balance the dataset
    if random_under_sample:
        rus = RandomUnderSampler(random_state=random_state, replacement=False)
        X, y = rus.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Perform one-hot encoding for categorical features
    categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_features)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_features)

    if shap_columns_to_drop is not None:
        # Drop non-important features from X_train and X_test
        X_train_encoded = X_train_encoded.drop(columns=shap_columns_to_drop)
        X_test_encoded = X_test_encoded.drop(columns=shap_columns_to_drop)

    return X_train_encoded, X_test_encoded, y_train, y_test


def xgb_model(X_train_encoded, X_test_encoded, y_train, y_test,
              param_grid: dict = None, scoring: str = 'f1', verbose: bool = True):
    """Runs predetermined xgboost algorithm on (preprocessed) Telco dataset. Returns the best model.
    :param """

    print(f'Running XGBoost model on dataset with {X_train_encoded.shape[1]} features with scoring metric {scoring}.')

    # Time-Logging
    start_time = time.time()

    # Define XGBoost classifier
    xgb_model = xgb.XGBClassifier()

    # Define hyperparameters grid for XGBoost
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [2, 3, 5],
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.75, 1],
            'colsample_bytree': [0.5, 0.75, 1]
        }

    if verbose:
        print(f'Parameter grid for grid search: {param_grid}')

    # Perform grid search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=10, n_jobs=-1, scoring=scoring)
    grid_search.fit(X_train_encoded, y_train)

    # Time-Logging
    end_time = time.time()
    print(f'XGBoost complete. Total training time: {round((end_time - start_time) / 60, 2)} minutes')

    if verbose:
        print(f'Best parameters: {grid_search.best_params_}')

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test_encoded)

    # Evaluate the model
    if verbose:
        conf_mat = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f"Classification Report: \n {classification_report(y_test, y_pred)}")
        print(f"Confusion matrix: TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp} \n", conf_mat)
        print(f'AUC score: {round(roc_auc_score(y_test, y_pred), 2)}')

    return best_model


def shap_analysis(xgb_model, X_test, min_normalized_importance: float = 0.1, verbose: bool = True):
    """Compute SHAP values for the test set and gets a feature ranking table. Returns columns to be dropped based on
    normalized_importance parameter."""

    # Time-Logging
    start_time = time.time()
    print(f'Running shapley analysis on dataset with {X_test.shape[1]} features.')

    if verbose:
        print(f'XGBoost model has parameters {xgb_model.get_params()}.\n')

    # Initialize an explainer with the best model's predict method
    explainer = shap.Explainer(xgb_model)
    # Compute SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # Time-Logging
    end_time = time.time()
    print(f'Total analysis time: {round((end_time - start_time) / 60, 2)} minutes.')

    # Visualize the SHAP summary plot
    if verbose:
        shap.summary_plot(shap_values, X_test)

    # Get feature ranking table and create normalized_importance feature
    feature_ranking = shapley_feature_ranking(X_test, shap_values)
    feature_ranking['normalized_importance'] = feature_ranking['importance'] / sum(feature_ranking['importance'])

    # Print 10 most important features
    if verbose:
        print("10 most important features, based on normalized importance:")
        print(feature_ranking.head(10))

    # select columns to be dropped due to low normalized_importance
    columns_to_drop = feature_ranking[feature_ranking["normalized_importance"] < min_normalized_importance]['features'].tolist()
    columns_to_keep = feature_ranking[feature_ranking["normalized_importance"] >= min_normalized_importance]['features'].tolist()

    print(f'Shapley analysis complete. Retained columns are: {columns_to_keep}')

    return columns_to_drop



def shapley_feature_ranking(X, shap_values):
    """Creates feature ranking table for X, based on shap_values"""
    feature_order = np.argsort(np.mean(np.abs(shap_values), axis=0))
    return pd.DataFrame(
        {
            "features": [X.columns[i] for i in feature_order][::-1],
            "importance": [
                np.mean(np.abs(shap_values), axis=0)[i] for i in feature_order
            ][::-1],
        }
    )