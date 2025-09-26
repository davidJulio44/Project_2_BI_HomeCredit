from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. Encode categorical features
    # First, identify categorical columns and their number of unique values
    categorical_columns = working_train_df.select_dtypes(include=['object']).columns
    binary_columns = []
    multi_category_columns = []
    
    for col in categorical_columns:
        n_unique = working_train_df[col].nunique()
        if n_unique == 2:
            binary_columns.append(col)
        else:
            multi_category_columns.append(col)
    
    # Handle binary categorical features using OrdinalEncoder
    if binary_columns:
        ordinal_encoder = OrdinalEncoder()
        working_train_df[binary_columns] = ordinal_encoder.fit_transform(working_train_df[binary_columns])
        working_val_df[binary_columns] = ordinal_encoder.transform(working_val_df[binary_columns])
        working_test_df[binary_columns] = ordinal_encoder.transform(working_test_df[binary_columns])
    
    # Handle multi-category features using OneHotEncoder
    if multi_category_columns:
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        onehot_train = onehot_encoder.fit_transform(working_train_df[multi_category_columns])
        onehot_val = onehot_encoder.transform(working_val_df[multi_category_columns])
        onehot_test = onehot_encoder.transform(working_test_df[multi_category_columns])
        
        # Get feature names for the encoded columns
        feature_names = onehot_encoder.get_feature_names_out(multi_category_columns)
        
        # Convert to DataFrame and join with original data
        onehot_train_df = pd.DataFrame(onehot_train, columns=feature_names, index=working_train_df.index)
        onehot_val_df = pd.DataFrame(onehot_val, columns=feature_names, index=working_val_df.index)
        onehot_test_df = pd.DataFrame(onehot_test, columns=feature_names, index=working_test_df.index)
        
        # Drop original columns and add encoded ones
        working_train_df = working_train_df.drop(multi_category_columns, axis=1).join(onehot_train_df)
        working_val_df = working_val_df.drop(multi_category_columns, axis=1).join(onehot_val_df)
        working_test_df = working_test_df.drop(multi_category_columns, axis=1).join(onehot_test_df)

    # 3. Impute missing values using median strategy
    imputer = SimpleImputer(strategy='median')
    working_train_df = pd.DataFrame(
        imputer.fit_transform(working_train_df),
        columns=working_train_df.columns,
        index=working_train_df.index
    )
    working_val_df = pd.DataFrame(
        imputer.transform(working_val_df),
        columns=working_val_df.columns,
        index=working_val_df.index
    )
    working_test_df = pd.DataFrame(
        imputer.transform(working_test_df),
        columns=working_test_df.columns,
        index=working_test_df.index
    )

    # 4. Scale features using Min-Max scaling
    scaler = MinMaxScaler()
    working_train_scaled = scaler.fit_transform(working_train_df)
    working_val_scaled = scaler.transform(working_val_df)
    working_test_scaled = scaler.transform(working_test_df)

    print("Output train data shape: ", working_train_scaled.shape)
    print("Output val data shape: ", working_val_scaled.shape)
    print("Output test data shape: ", working_test_scaled.shape)

    return working_train_scaled, working_val_scaled, working_test_scaled
