import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def general_description(df: pd.DataFrame) -> None:
    """
    General description of the dataset.

    Args:
        df (pd.DataFrame): The pandas DataFrame to describe.

    Returns:
        None
    """
    print("\nGeneral Description:\n") 
    print("Number of records: ", len(df))
    print("\nNumber of columns: ", len(df.columns))
    print("\nNumber of Null records:\n", df.isna().sum())
    print("\nNumber of Duplicate records: ", df.duplicated().sum())
    print("\nData types of Variables:\n", df.dtypes)
    print("\nGeneral Description of Numeric Variables:\n", df.describe())

def null_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Null treatment function.

    Args:
        df (pd.DataFrame): The pandas DataFrame for null treatment.

    Returns:
        pd.DataFrame: DataFrame after null treatment.
    """
    print("\nNull Treatment:\n")
    print("\nNumber of records before null treatment: ", df.shape)

    # Filling missing values in GENDER with mode
    df['GENDER'] = df['GENDER'].fillna(df['GENDER'].mode()[0])

    print("\nTreatment for Gender: ", df.shape)

    # Dropping duplicate records
    df = df.drop_duplicates()
    print("\nNumber of records after removing duplicates: ", df.shape)

    # Removing records where NET_SALES is less than 0
    df = df[df['NET_SALES'] > 0]
    print("\nNumber of records after validation for Net Sales > 0: ", df.shape)
    
    # Removing records where DURATION is less than 0
    df = df[df['DURATION'] > 0]
    print("\nNumber of records after validation for Duration > 0: ", df.shape)

    # Removing records where AGE is less than 0 or greater than or equal to 100
    df = df[(df['AGE'] > 0) & (df['AGE'] < 100)]
    print("\nNumber of records after validation for 0 < Age < 100: ", df.shape)    
    
    print("\nNumber of Null records:\n", df.isna().sum())
    
    return df

def assign_features(df: pd.DataFrame, y_name: str) -> tuple:
    """
    Assign features function.

    Args:
        df (pd.DataFrame): The pandas DataFrame to extract features from.
        y_name (str): The name of the target variable.

    Returns:
        pd.DataFrame: Features DataFrame (X_df)
        pd.DataFrame: Target DataFrame (y_df)
        pd.DataFrame: Categorical variables DataFrame (cat_var_df)
        pd.DataFrame: Numeric variables DataFrame (num_var_df)
        list: List of categorical variable names (cat_var_names)
        list: List of numeric variable names (num_var_names)
    """
    X_df = df.drop(columns=y_name) 
    y_df = df[y_name] 
    cat_var_df = X_df.select_dtypes(include=['object', 'category', 'bool'])
    num_var_df = X_df.select_dtypes(include=['number'])
    cat_var_names = cat_var_df.columns.tolist()
    num_var_names = num_var_df.columns.tolist()
    return X_df, y_df, cat_var_df, num_var_df, cat_var_names, num_var_names
