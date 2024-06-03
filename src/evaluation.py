# Imports
import numpy as np
import pandas as pd
import json

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score


def compare_model_results(full_X_test=None, shap_X_test=None, sr_X_test=None, y_test=None,
                          full_xgb_model=None, shap_xgb_model=None, sr_model=None):
    """ Create an evaluation overview of all available models."""

    if full_xgb_model is not None:
        y_pred = full_xgb_model.predict(full_X_test)
        create_evaluation_output("full XGBoost", y_test, y_pred)

    if shap_xgb_model is not None:
        y_pred = shap_xgb_model.predict(shap_X_test)
        create_evaluation_output("Shapley XGBoost", y_test, y_pred)


def evaluate_SR_expression(result_path, test_data_path, threshold):
    """Evaluate the Symbolic Regression expression"""

    # load best_expression and df
    with open(result_path, 'r') as file:
        data = json.load(file)

    expressions = []
    for i in range(len(data)):
        expressions.append(data[i]["test"]["best_expr"])

    df = pd.read_csv(test_data_path, header=None, index_col=None)

    # Define the safe evaluation environment
    safe_env = {
        'df': df,
        'sin': np.sin,
        'cos': np.cos,
        'log': np.log,
        'exp': np.exp,
        'sqrt': np.sqrt,
    }

    # Evaluate each expression
    for exp in expressions:

        # Replace variables with dataframe column references
        expression = exp.replace("A", "df.iloc[:, 0]").replace("B", "df.iloc[:, 1]").replace("D", "df.iloc[:, 2]")

        # Use eval to predict
        y_test = df.iloc[:, -1].values
        y_pred = eval(expression, {"__builtins__": None}, safe_env)
        y_pred_binarized = (y_pred > threshold).astype(int)

        # Evaluate the model
        create_evaluation_output("Symbolic Regression", y_test, y_pred_binarized,
                                 expression=exp)


def create_evaluation_output(model_name: str, y_test, y_pred, expression: str = None):
    conf_mat = confusion_matrix(y_test, y_pred)
    if expression is None:
        print(f"Evaluation of the {model_name} model: \n {conf_mat}")
    else:
        print(f"Evaluation of the {model_name} model with expression {expression}: \n {conf_mat}")
    # print(f"- Classification report: {classification_report(y_test, y_pred, output_dict=True)}")
    print(f'- AUC score: {round(roc_auc_score(y_test, y_pred), 2)}')
    print(f'- Precision: {round(precision_score(y_test, y_pred), 2)}')
    print(f'- Recall: {round(recall_score(y_test, y_pred), 2)}')
    print(f'- F1 score: {round(f1_score(y_test, y_pred), 2)}')
    print('\n')

