# Imports
from argparse import ArgumentParser
from preprocessing import load_Telco, preprocess_Telco, load_and_preprocess_KKBox
from applied_xgboost import create_xgb_train_test, xgb_model, shap_analysis
from evaluation import compare_model_results



if __name__ == '__main__':
    parser = ArgumentParser(prog='XGB', description='Run XGBoost with SHAP analysis on dataset.')
    parser.add_argument("-config", default="../configs/test_config.json")
    args = parser.parse_args()

    # Fixed Settings
    random_state = 42
    test_size = 0.1
    min_normalized_importance = 0.1
    verbose = False
    dataset = 'KKBox'

    # Variable Settings
    if dataset == 'Telco':
        drop_x = ['customerID','Churn']
        target_name = 'Churn'
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [2, 3, 5],
            'learning_rate': [ 0.1, 0.01, 0.001],
            'subsample': [0.5, 0.75, 1],
            'colsample_bytree': [0.5, 0.75, 1]
        }

        data = load_Telco(path='./data/Telco-Customer-Churn.csv')
        data = preprocess_Telco(data)

    elif dataset == 'KKBox':
        drop_x = ['msno','is_churn', 'city', 'registered_via']
        target_name = 'is_churn'
        param_grid = {
            'n_estimators': [300],
            'max_depth': [10],
            'learning_rate': [0.01],
        }

        data = load_and_preprocess_KKBox(path_to_folder='./data/kkbox/')

    else:
        raise ValueError(f"Wrong dataset name '{dataset}'. Allowed names are 'Telco', 'KKBox'.")



    # Create training and test set
    full_X_train, full_X_test, y_train, y_test = create_xgb_train_test(data, drop_x=drop_x, target_name=target_name,
                                                             test_size=test_size, random_state=random_state)

    # Run XGBoost algorithm
    full_xgb_model = xgb_model(full_X_train, full_X_test, y_train, y_test,
                               param_grid=param_grid,
                               verbose=verbose)

    # Perform Shap analysis
    columns_to_drop = shap_analysis(full_xgb_model, full_X_test,
                                    min_normalized_importance=min_normalized_importance,
                                    verbose=verbose)

    # Create training and test set considering Shap analysis
    shap_X_train, shap_X_test, y_train, y_test = create_xgb_train_test(data,
                                                                       drop_x=drop_x,
                                                                       target_name=target_name,
                                                                       test_size=test_size,
                                                                       random_state=random_state,
                                                                       shap_columns_to_drop=columns_to_drop)

    # Run XGBoost algorithm
    shap_xgb_model = xgb_model(shap_X_train, shap_X_test, y_train, y_test,
                               param_grid=param_grid,
                               verbose=verbose)


    # Evaluation
    compare_model_results(full_X_test=full_X_test, shap_X_test=shap_X_test, sr_X_test=None,
                          y_test=y_test,
                          full_xgb_model=full_xgb_model,
                          shap_xgb_model=shap_xgb_model,
                          sr_model=None)




# TODO: Symbolic Regression Pipeline

# Pipeline Input
# - Dataset
# - Size of Dataset
# - Height of Tree
# - max function complexity?
# - function with constant or without

# Pipeline Structure
# - create expression set if not available
# - create train parameters if not available
# - run symbolic regression
# - automatic function conversion
# - evaluation of the SR function
# - Probability effect of each function component

