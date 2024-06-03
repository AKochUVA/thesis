# Imports
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score


def compare_model_results(full_X_test=None, shap_X_test=None, sr_X_test=None, y_test=None,
                          full_xgb_model=None, shap_xgb_model=None, sr_model=None):
    """ Create a evaluation overview of all available models."""

    if full_xgb_model is not None:
        y_pred = full_xgb_model.predict(full_X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        print(f"Evaluation of the full XGBoost model: \n {conf_mat}")
        #print(f"- Classification report: {classification_report(y_test, y_pred, output_dict=True)}")
        print(f'- AUC score: {round(roc_auc_score(y_test, y_pred), 2)}')
        print(f'- Precision: {round(precision_score(y_test, y_pred), 2)}')
        print(f'- Recall: {round(recall_score(y_test, y_pred), 2)}')
        print(f'- F1 score: {round(f1_score(y_test, y_pred), 2)}')

    if shap_xgb_model is not None:
        y_pred = shap_xgb_model.predict(shap_X_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        print(f"Evaluation of the Shapley XGBoost model: \n {conf_mat}")
        #print(f"- Classification report: {classification_report(y_test, y_pred, output_dict=True)}")
        print(f'- AUC score: {round(roc_auc_score(y_test, y_pred), 2)}')
        print(f'- Precision: {round(precision_score(y_test, y_pred), 2)}')
        print(f'- Recall: {round(recall_score(y_test, y_pred), 2)}')
        print(f'- F1 score: {round(f1_score(y_test, y_pred), 2)}')

