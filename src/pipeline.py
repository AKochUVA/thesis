# Imports
import time
from argparse import ArgumentParser
from preprocessing import load_Telco, preprocess_Telco, load_and_preprocess_KKBox
from applied_xgboost import create_xgb_train_val_test, xgb_model, shap_analysis
from evaluation import compare_model_results, evaluate_SR_expression
from utils import (load_config_file, generate_sr_expressions, train_HVAE_model, run_symbolic_classification,
                   save_train_val_test_data)

if __name__ == '__main__':
    parser = ArgumentParser(prog='Pipeline', description='Full XGBoost w/ Shapley + Symbolic Regression pipeline.')
    parser.add_argument("-config", default="./configs/test_config.json")
    args = parser.parse_args()


    # Time-keeping
    start_time = time.time()

    # Load configurations
    config = load_config_file(args.config)
    expr_def_config = config['expression_definition']
    expr_gen_config = config['expression_set_generation']
    training_config = config['training']
    symbolic_regression_config = config['symbolic_regression']

    # Fixed Settings
    random_state = 2
    val_size = 0.1
    test_size = 0.1

    # normalized importance determines number of variables kept
    #min_normalized_importance = 0.1

    # the number of variables kept can also be selected directly
    num_shap_features = 20

    use_shapley_values = True
    use_existing_expression_set = False
    use_existing_HVAE_parameters = False
    verbose = True
    dataset = 'kkbox'  # 'telco' or 'kkbox'

    # Variable Settings
    if dataset == 'telco':
        drop_x = ['customerID', 'Churn']
        target_name = 'Churn'
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [2, 3, 5],
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.75, 1],
            'colsample_bytree': [0.5, 0.75, 1]
        }

        data = load_Telco(path='./data/telco/Telco-Customer-Churn.csv')
        data = preprocess_Telco(data)

    elif dataset == 'kkbox':
        drop_x = ['msno', 'is_churn', 'city', 'registered_via']
        target_name = 'is_churn'
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.001, 0.01],
            'subsample': [0.75, 1],
            'colsample_bytree': [0.75, 1]
        }

        data = load_and_preprocess_KKBox(path_to_folder='./data/kkbox/')

    else:
        raise ValueError(f"Wrong dataset name '{dataset}'. Allowed names are 'telco', 'kkbox'.")

    # Create training, validation and test set
    full_X_train, full_X_val, full_X_test, y_train, y_val, y_test = create_xgb_train_val_test(data,
                                                                                              drop_x=drop_x,
                                                                                              target_name=target_name,
                                                                                              val_size=val_size,
                                                                                              test_size=test_size,
                                                                                              random_state=random_state)

    print(f"Size of training set: {len(full_X_train)}. \n"
          f"Size of validation set: {len(full_X_val)}. \n"
          f"Size of test set: {len(full_X_test)}.")

    # Run XGBoost algorithm, evaluated on validation set
    full_xgb_model = xgb_model(full_X_train, full_X_val, y_train, y_val,
                               param_grid=param_grid,
                               verbose=verbose)

    if use_shapley_values:
        # Perform Shap analysis
        columns_to_drop = shap_analysis(full_xgb_model, full_X_test,
                                        #min_normalized_importance=min_normalized_importance,
                                        num_shap_features=num_shap_features,
                                        verbose=verbose)

        # Create training, validation and test set considering Shap analysis
        shap_X_train, shap_X_val, shap_X_test, y_train, y_val, y_test = create_xgb_train_val_test(data,
                                                                                                  drop_x=drop_x,
                                                                                                  target_name=target_name,
                                                                                                  val_size=val_size,
                                                                                                  test_size=test_size,
                                                                                                  random_state=random_state,
                                                                                                  shap_columns_to_drop=columns_to_drop)

        # Run XGBoost algorithm
        shap_xgb_model = xgb_model(shap_X_train, shap_X_val, y_train, y_val,
                                   param_grid=param_grid,
                                   verbose=verbose)

        # Create training, validation and test set as .csv files for Symbolic Regression
        train_set_path, val_set_path, test_set_path = save_train_val_test_data(shap_X_train, shap_X_val, shap_X_test,
                                                                               y_train, y_val, y_test,
                                                                               dataset_name=dataset,
                                                                               random_state=random_state)
        num_variables = len(shap_X_train.columns)
    else:
        train_set_path, val_set_path, test_set_path = save_train_val_test_data(full_X_train, full_X_val, full_X_test,
                                                                               y_train, y_val, y_test,
                                                                               dataset_name=dataset,
                                                                               random_state=random_state)
        num_variables = len(full_X_train.columns)

    # Set maximum tree height
    maximum_tree_height = min(int(num_variables * 2 + 1), 8)  # 10 is the maximum allowed maximum tree height

    # SR Time-keeping
    sr_start_time = time.time()

    # Generate Expression Set
    expression_set_path = generate_sr_expressions(symbols=expr_def_config['symbols'],
                                                  num_variables=num_variables,
                                                  has_constants=expr_def_config['has_constants'],
                                                  num_expressions=expr_gen_config['num_expressions'],
                                                  max_tree_height=maximum_tree_height,
                                                  expression_set_path=expr_gen_config['expression_set_path'],
                                                  filename=None,
                                                  use_existing=use_existing_expression_set)

    # Train the HVAE model
    params_path = train_HVAE_model(symbols=expr_def_config['symbols'],
                                   num_variables=num_variables,
                                   has_constants=expr_def_config['has_constants'],
                                   max_tree_height=maximum_tree_height,
                                   expression_set_path=expression_set_path,
                                   training_config=training_config,
                                   verbose=verbose,
                                   filename=None,
                                   use_existing=use_existing_HVAE_parameters,
                                   random_state=random_state)

    # Run Symbolic Classification and select best model on Validation Set
    results_path = run_symbolic_classification(config, symbolic_regression_config, num_variables=num_variables,
                                               symbols=expr_def_config['symbols'],
                                               has_constants=expr_def_config['has_constants'],
                                               max_tree_height=maximum_tree_height, params_path=params_path,
                                               train_set_path=train_set_path, val_set_path=val_set_path,
                                               dataset_name=dataset, random_state=random_state)

    # SR Time-keeping
    print(f"Total Symbolic Regression Computation Time: {round((time.time() - sr_start_time) / 60, 2)} minutes.")

    # Evaluation of all Models on Test Set
    compare_model_results(full_X_test=full_X_test,
                          shap_X_test=shap_X_test if use_shapley_values else None,
                          sr_X_test=None,
                          y_test=y_test,
                          full_xgb_model=full_xgb_model,
                          shap_xgb_model=shap_xgb_model if use_shapley_values else None,
                          sr_model=None)

    col_names = shap_X_test.columns if use_shapley_values else full_X_test.columns

    evaluate_SR_expression(results_path, test_set_path, symbolic_regression_config['threshold'],
                           num_vars=num_variables, column_names=col_names)

    # Time-keeping
    print(f"Total execution time: {round((time.time() - start_time) / 60, 2)} minutes.")

