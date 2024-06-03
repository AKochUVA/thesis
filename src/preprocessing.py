# Imports
import os
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
import tqdm


def load_Telco(path: str):
    """Loads the Telco dataset with NA values as empty strings."""
    data = pd.read_csv(filepath_or_buffer=path, na_values=[' '])
    return data


def preprocess_Telco(data):
    """Preprocesses the Telco dataset.
    Preprocessing steps are:
    - Drop NaN rows
    - Map gender column to binary values
    - Map "No phone service" to "No"
    - Map "No internet service" to "No"
    - Map No to 0 and Yes to 1
    - Change TotalCharges dtype
    - Reformat PaymentMethod column
    - Reformat Contract column
    - Reformat InternetService column

    """

    # Drop NaN rows
    data = data.dropna(axis=0)

    # Map gender column to binary values
    gender_mapping = {'Male': 1, 'Female': 0}
    data['gender'] = data['gender'].map(lambda x: 1 if x == 'Male' else 0)

    # Map "No phone service" to "No"
    data['MultipleLines'] = data['MultipleLines'].map(lambda x: 'No' if x == 'No phone service' else x)

    # Map "No internet service" to "No"
    internet_services_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                 'StreamingMovies']
    data[internet_services_columns] = data[internet_services_columns].map(
        lambda x: "No" if x == "No internet service" else x)

    # Map No to 0 and Yes to 1
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'] + internet_services_columns
    data[yes_no_columns] = data[yes_no_columns].map(lambda x: 0 if x == "No" else 1)

    # Change TotalCharges dtype
    data['TotalCharges'] = data['TotalCharges'].astype(float)

    # Reformat PaymentMethod column
    payment_method_mapping = {'Bank transfer (automatic)': 'BankTransfer', 'Credit card (automatic)': 'CreditCard',
                              'Electronic check': 'ElectronicCheck', 'Mailed check': 'MailedCheck'}
    data['PaymentMethod'] = data['PaymentMethod'].replace(payment_method_mapping)

    # Reformat Contract column
    contract_mapping = {'Month-to-month': 'MonthToMonth', 'One year': 'OneYear', 'Two year': 'TwoYear'}
    data['Contract'] = data['Contract'].map(contract_mapping)

    # Reformat InternetService column
    internet_service_mapping = {'Fiber optic': 'FiberOptic', 'DSL': 'DSL'}
    data['InternetService'] = data['InternetService'].map(internet_service_mapping)

    return data

def load_and_preprocess_KKBox(path_to_folder):
    """Loads and preprocesses KKBox datasets and returns them as a single pandas dataframe."""

    # Data Paths
    train_path = os.path.join(path_to_folder, 'train.csv')
    members_path = os.path.join(path_to_folder, 'members_v3.csv')
    transaction_features_path = os.path.join(path_to_folder, 'transactions_features.csv')
    user_logs_features_path = os.path.join(path_to_folder, 'user_logs_features2.csv')

    # Load Datasets
    train = pd.read_csv(train_path)
    members = pd.read_csv(members_path)
    transaction_features = pd.read_csv(transaction_features_path, index_col=0)
    user_logs_features = pd.read_csv(user_logs_features_path, index_col=0)

    # Pre-processing for Members Data (pre-processing for transactions and user_logs happened externally due to size)
    # TODO: add link to pre-processing for transactions and user_logs
    # bd renamed as age
    members = members.rename(columns={'bd': 'age'})

    # members with ages outside of 0-99 can be dropped
    # 4.5mn members (out of 6.7mn total) have age 0 (i.e. using the current date as their birthday), they can not be dropped
    members = members[(members['age'] >= 0) & (members['age'] < 99)]

    # replace NAs in gender
    members['gender'] = members['gender'].fillna('not_specified')

    # registration_init_time as datetime
    members['registration_init_time'] = pd.to_datetime(members['registration_init_time'].astype('str'), format='%Y%m%d')
    # convert to days since unix epoch start
    members['registration_init_time'] = members['registration_init_time'].astype('int') / (10 ** 9 * 60 * 60 * 24)
    # convert to days since first registration
    members['registration_init_time'] = members['registration_init_time'] - min(members['registration_init_time'])

    # create categorical columns for city, age, gender and registered_via
    members['city'] = members['city'].astype('category')
    # members['age'] = members['age'].astype('category')
    members['gender'] = members['gender'].astype('category')
    members['registered_via'] = members['registered_via'].astype('category')

    # Create singular train dataset
    data = pd.merge(train, members, on='msno', how='left')
    data = pd.merge(data, transaction_features, on='msno', how='left')
    data = pd.merge(data, user_logs_features, on='msno', how='left')

    return data
