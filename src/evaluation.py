# Imports
import numpy as np
import pandas as pd
import json

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt


def compare_model_results(full_X_test=None, shap_X_test=None, sr_X_test=None, y_test=None,
                          full_xgb_model=None, shap_xgb_model=None, sr_model=None):
    """ Create an evaluation overview of all available models."""

    if full_xgb_model is not None:
        y_pred = full_xgb_model.predict(full_X_test)
        create_evaluation_output("Full XGBoost", y_test, y_pred_binarized=y_pred)

    if shap_xgb_model is not None:
        y_pred = shap_xgb_model.predict(shap_X_test)
        create_evaluation_output("Shapley XGBoost", y_test, y_pred_binarized=y_pred)


def evaluate_SR_expression(result_path, test_data_path, threshold, num_vars, column_names):
    """Evaluate the Symbolic Regression expression"""

    if num_vars <= 0:
        raise ValueError("num_vars must be greater than 0.")

    # load best_expressions, constants and df
    with open(result_path, 'r') as file:
        data = json.load(file)

    expressions = []
    for i in range(len(data)):
        expressions.append(data[i]["test"]["best_expr"])

    constants = []

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

        # create copy
        expression = exp

        # List of valid characters except 'C'
        valid_chars = [chr(i) for i in range(65, 91) if chr(i) != 'C']  # 65-90 are ASCII values for A-Z

        # Replace variables with dataframe column references in a loop
        for i in range(num_vars):
            char = valid_chars[i]
            expression = expression.replace(char, f"df.iloc[:, {i}]")

        # Replace constants C1, C2, C3, etc. with their values from the list of constants
        for i, value in enumerate(constants, start=1):
            expression = expression.replace(f"C{i}", str(value))

        # Replace ^2, ^3, etc. with **2, **3, etc.
        for power in range(2, 6):
            expression = expression.replace(f"^{power}", f"**{power}")

        # Use eval to predict
        y_test = df.iloc[:, -1].values
        y_pred = sigmoid(eval(expression, {"__builtins__": None}, safe_env))
        y_pred_binarized = (y_pred > threshold).astype(int)

        # Calculate marginal contributions
        try:
            average_effect = calculate_average_effect(expression, valid_chars, num_vars, safe_env)
        except:
            average_effect = None

        # Calculate base model prediction (i.e. all variables = 0)
        try:
            base_model_prediction = calculate_base_model_prediction(expression, valid_chars, num_vars, safe_env)
        except ZeroDivisionError:
            # For manual calculation
            base_model_prediction = sigmoid(0-0-1/18)
            print("INSERTED BASE PREDICTION MANUALLY!")

        # Evaluate the model
        create_evaluation_output("Symbolic Regression", y_test, y_pred, y_pred_binarized, threshold,
                                 base_model_prediction, average_effect, expression=exp,
                                 column_names=column_names)




def create_evaluation_output(model_name: str, y_test, y_pred=None, y_pred_binarized=None,
                             threshold=0.5, base_model_prediction: float = None, average_effects: dict = None,
                             expression: str = None, column_names: list = None) -> None:
    conf_mat = confusion_matrix(y_test, y_pred_binarized)
    if expression is None:
        # XGBoost models use binary predictions only
        print(f"Evaluation of the {model_name} model: \n {conf_mat}")
        print(f'- AUC score: {round(roc_auc_score(y_test, y_pred_binarized), 2)}')
        print(f'- Precision: {round(precision_score(y_test, y_pred_binarized), 2)}')
        print(f'- Recall: {round(recall_score(y_test, y_pred_binarized), 2)}')
        print(f'- F1 score: {round(f1_score(y_test, y_pred_binarized), 2)}')
    else:
        print(f"Evaluation of the {model_name} model with expression {expression}: \n {conf_mat}")
        print(f"Expression translates to: {translate_expression(expression, column_names)}")
        # print(f"- Classification report: {classification_report(y_test, y_pred, output_dict=True)}")
        print(f'- AUC score: {round(roc_auc_score(y_test, y_pred), 2)}')
        print(f'- Precision: {round(precision_score(y_test, y_pred_binarized), 2)}')
        print(f'- Recall: {round(recall_score(y_test, y_pred_binarized), 2)}')
        print(f'- F1 score: {round(f1_score(y_test, y_pred_binarized), 2)}')
        print(f'Base model prediction (all variables = 0): {base_model_prediction}')
        print("Average feature effects:", average_effects, " Please be advised that these values are approximate"
                                                                 " and are most likely incorrect when the expression "
                                                                 "contains non-linear interactions between variables.")
        print('\n')

        #create_histogram(base_model_prediction, threshold, y_pred, y_test)


def create_histogram(base_model_prediction, threshold, y_pred, y_test):
    # Create Histogram
    plt.figure(figsize=(6, 4))
    df = pd.DataFrame({'y_pred': y_pred, 'y_true': y_test})
    ax = sns.histplot(data=df, x=y_pred, bins=50, hue=y_test, multiple="stack")
    plt.legend(loc='upper right', labels=[1, 0], title='True Value', fontsize=9, title_fontsize=9)
    # Vertical line for decision threshold
    ax.axvline(threshold, color='r', linestyle='--')
    plt.text(threshold + 0.01, ax.get_ylim()[1] * 0.9, f't={threshold}', color='red', fontsize=9, va='top',
             ha='left')
    if base_model_prediction is not None:
        # Vertical line for base model prediction
        ax.axvline(base_model_prediction, color='blue', linestyle='--')
        plt.text(base_model_prediction - 0.01, ax.get_ylim()[1] * 0.9,
                 f'base\nmodel\nprediction\n={round(base_model_prediction, 2)}', color='blue', fontsize=9, va='top',
                 ha='right')
    plt.xlim((0, 1))
    # Add labels and title
    plt.xlabel('Predicted Value')
    plt.ylabel('Count')
    # plt.title(f'{translate_expression(expression, column_names)}\n'
    #           f'AUC: {round(roc_auc_score(y_test, y_pred), 2)}, F1: {round(f1_score(y_test, y_pred_binarized),2)}',
    #           fontsize=8)

    plt.show()


def translate_expression(expression, column_names):
    """Use the column names in the test set to translate the expression to a readable format."""
    # List of valid characters except 'C'
    valid_chars = [chr(i) for i in range(65, 91) if chr(i) != 'C']  # 65-90 are ASCII values for A-Z

    # Replace variables with dataframe column references in a loop
    for i in range(len(column_names)):
        char = valid_chars[i]
        expression = expression.replace(char, str(column_names[i]))

    return expression


def calculate_average_effect(expression, valid_chars, num_vars, safe_env):
    """Calculate the average effect of each variable by setting it to 1 or 0."""
    average_effects = {}

    for i in range(num_vars):
        char = valid_chars[i]

        # Evaluate with the variable set to 1
        expression_1 = expression.replace(f"df.iloc[:, {i}]", "1")
        y_pred_1 = sigmoid(eval(expression_1, {"__builtins__": None}, safe_env))

        # Evaluate with the variable set to 0
        expression_0 = expression.replace(f"df.iloc[:, {i}]", "0")
        y_pred_0 = sigmoid(eval(expression_0, {"__builtins__": None}, safe_env))

        # Calculate the average effect
        average_effect = np.mean(y_pred_1 - y_pred_0)
        average_effects[char] = round(average_effect, 2)

    return average_effects


def calculate_base_model_prediction(expression, valid_chars, num_vars, safe_env):
    """Calculate the prediction of the expression when all variables are set to 0"""

    # set all variables to 0
    for i in range(num_vars):
        expression = expression.replace(f"df.iloc[:, {i}]", "0")

    # Evaluate the expression
    base_pred = sigmoid(eval(expression, {"__builtins__": None}, safe_env))


    return round(base_pred,2)

def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    dataset = 'kkbox'
    num_vars = 3
    depth = min(num_vars * 2 + 1, 11)
    seed = 2

    results_path = f'../results/{dataset}_{num_vars}var_{depth}depth_seed{seed}.json'
    test_set_path = f'../data/train_val_test_data/{dataset}_{num_vars}vars_test_seed{seed}.csv'

    telco_column_names2 = ['InternetService_FiberOptic', 'Contract_MonthToMonth']
    telco_column_names3 = ['Tenure', 'InternetService_FiberOptic', 'Contract_MonthToMonth']
    telco_column_names5 = ['Tenure', 'InternetService_FiberOptic', 'Contract_MonthToMonth',
                           'Contract_TwoYear', 'PaymentMethod_ElectronicCheck']
    telco_column_names5_seed2 = ['tenure', 'MonthlyCharges', 'TotalCharges', 'InternetService_FiberOptic',
                                 'Contract_MonthToMonth']
    telco_column_names10_seed0 = ['tenure', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
                                  'InternetService_FiberOptic',
                                  'Contract_MonthToMonth', 'Contract_OneYear', 'Contract_TwoYear',
                                  'PaymentMethod_ElectronicCheck']

    telco_column_names10_seed1 = ['tenure', 'OnlineSecurity', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
                                  'InternetService_DSL',
                                  'InternetService_FiberOptic', 'Contract_MonthToMonth', 'Contract_TwoYear',
                                  'PaymentMethod_ElectronicCheck']

    telco_column_names10_seed2 = ['tenure', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
                                  'InternetService_DSL',
                                  'InternetService_FiberOptic', 'Contract_MonthToMonth', 'Contract_TwoYear',
                                  'PaymentMethod_ElectronicCheck']

    if dataset == 'telco':
        if num_vars == 2:
            column_names = telco_column_names2
        elif num_vars == 3:
            column_names = telco_column_names3
        elif num_vars == 5:
            if seed == 2:
                column_names = telco_column_names5_seed2
            else:
                column_names = telco_column_names5
        elif num_vars == 10:
            if seed == 0:
                column_names = telco_column_names10_seed0
            elif seed == 1:
                column_names = telco_column_names10_seed1
            elif seed == 2:
                column_names = telco_column_names10_seed2
    elif dataset == 'kkbox':
        if num_vars == 3:
            if seed == 0 or seed == 2:
                column_names = ['mean_auto_renew', 'last_auto_renew', 'last_cancel']
            elif seed == 1:
                column_names = ['num_transactions', 'last_auto_renew', 'last_cancel']
        elif num_vars == 5:
            if seed == 0:
                column_names = ['num_transactions', 'mean_auto_renew', 'last_auto_renew', 'mean_cancel', 'last_cancel']
            elif seed == 1:
                column_names = ['num_transactions', 'mean_act_pay', 'mean_auto_renew', 'last_auto_renew', 'last_cancel']

    evaluate_SR_expression(results_path, test_set_path, threshold=0.4, num_vars=num_vars, column_names=column_names)