import pandas as pd


def evaluate_results(path, func, threshold):
    """Function to evaluate the classification accuracy using the generated expression."""
    # Import test data
    test_data = pd.read_csv(path)
    y_test = test_data.iloc[:, -1].values

    y_pred = test_data.apply(func)
    y_pred_binarized = (y_pred > threshold).astype(int)

def symbolic_regression_function(df):
    A = 'tenure'
    B = 'InternetService_FiberOptic'
    D = 'Contract_MonthToMonth'
    C1 = -0.6965971995944867
    C2 = 1.2781484814751356
    C3 = 1.2430512333536776
    C4 = 0.027783268153546238
    C5 = -2.469009073022163

    return (((C1*C2+C3*df[D])-df[B])-df[A]*C4)-C5*df[B]