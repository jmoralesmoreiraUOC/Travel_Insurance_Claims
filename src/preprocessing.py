from typing import List, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder, LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectPercentile, SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from boruta import BorutaPy

def num_var_df_transform(num_var_df: pd.DataFrame, num_var_names: List[str]) -> pd.DataFrame:
    """
    Function to transform numerical variables in a DataFrame using PowerTransformer.

    Args:
        num_var_df (pd.DataFrame): DataFrame containing numerical variables.
        num_var_names (List[str]): List of numerical variable names.

    Returns:
        pd.DataFrame: DataFrame with transformed numerical variables.
    """
    # Initializing PowerTransformer
    pt = PowerTransformer()

    # Transforming each numerical variable
    for col in num_var_names:
        col_tf = pt.fit_transform(num_var_df[[col]])
        col_tf = np.array(col_tf).reshape(col_tf.shape[0])  # Reshaping to match DataFrame structure
        num_var_df[col] = col_tf
    
    return num_var_df

def treat_outliers_truncation(data: pd.DataFrame, col: str) -> np.ndarray:
    """
    Function to treat outliers using truncation method.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        col (str): Column name for which outliers will be treated.

    Returns:
        np.ndarray: Array with outliers treated using truncation.
    """
    # Calculating lower and upper limits
    lower_limit, upper_limit = data[col].quantile([0.25, 0.75])
    
    # Calculating interquartile range (IQR)
    IQR = upper_limit - lower_limit
    
    # Calculating lower and upper whiskers
    lower_whisker = lower_limit - 1.5 * IQR
    upper_whisker = upper_limit + 1.5 * IQR
    
    # Truncating outliers
    return np.where(data[col] > upper_whisker, upper_whisker,
                    np.where(data[col] < lower_whisker, lower_whisker, data[col]))

def treat_outliers_smote(df: pd.DataFrame, col: str, Y_NAME: str) -> pd.Series:
    """
    Function to treat outliers using SMOTE method.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Column name for which outliers will be treated.
        Y_NAME (str): Target variable name.

    Returns:
        pd.Series: Series with outliers treated using SMOTE.
    """
    # Convert target variable to categorical if not already
    if df[Y_NAME].dtype != 'object' and df[Y_NAME].dtype != 'category':
        df[Y_NAME] = df[Y_NAME].astype('category')

    # Select only non-outliers data for synthetic imputation
    non_outliers = df[df[col].between(df[col].quantile(0.25), df[col].quantile(0.75))]

    # Apply SMOTE to generate synthetic outliers
    smote = SMOTE(sampling_strategy='minority')
    synthetic_outliers, synthetic_labels = smote.fit_resample(non_outliers[[col]], non_outliers[Y_NAME])

    # Replace original outliers with synthetic values
    df.loc[~df.index.isin(non_outliers.index), col] = synthetic_outliers

    return df[col]

def target_encoded(df: pd.DataFrame, Y_CLAIM: str) -> pd.DataFrame:
    """
    Function to encode the target variable using LabelEncoder.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        Y_CLAIM (str): Name of the target variable.

    Returns:
        pd.DataFrame: DataFrame with the target variable encoded.
    """
    # Encoding the target variable using LabelEncoder
    label_encoder = LabelEncoder()
    df[Y_CLAIM] = label_encoder.fit_transform(df[Y_CLAIM])
    
    return df

def cat_var_labelEncoder(df: pd.DataFrame, cat_var_label_encoder: List[str]) -> pd.DataFrame:
    """
    Function to encode categorical variables using LabelEncoder.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_var_label_encoder (List[str]): List of column names for categorical variables.

    Returns:
        pd.DataFrame: DataFrame with categorical variables encoded.
    """
    # Encoding each categorical variable using LabelEncoder
    for cat in cat_var_label_encoder:
        label_encoder = LabelEncoder()
        df[cat] = label_encoder.fit_transform(df[cat])
    
    return df

def cat_var_ordinalEncoder(df: pd.DataFrame, cat_var_ordinal_encoder: List[str]) -> pd.DataFrame:
    """
    Function to encode categorical variables using OrdinalEncoder.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_var_ordinal_encoder (List[str]): List of column names for categorical variables.

    Returns:
        pd.DataFrame: DataFrame with categorical variables encoded.
    """
    # Encoding each categorical variable using OrdinalEncoder
    for cat in cat_var_ordinal_encoder:
        categories_unique = df[cat].unique().tolist()
        ordinal_encoder = OrdinalEncoder(categories=[categories_unique])
        df[cat] = ordinal_encoder.fit_transform(df[[cat]])
        df[cat] = df[cat].astype(int)
    
    return df     

def categorize_commission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to categorize commission values into low, medium, and high.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        pd.DataFrame: DataFrame with commission values categorized.
    """
    def categorize_commission_fun(value):
        if value >= 0 and value <= 87:
            return "Low"
        elif value > 87 and value <= 174:
            return "Medium"
        else:
            return "High"
    
    # Applying commission categorization
    df['COMMISION'] = df['COMMISION'].apply(categorize_commission_fun)

    return df

def onehot_encoder(df: pd.DataFrame, cat_var_onehot_encoder: List[str]) -> pd.DataFrame:
    """
    Function to encode categorical variables using one-hot encoding.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_var_onehot_encoder (List[str]): List of column names for categorical variables.

    Returns:
        pd.DataFrame: DataFrame with categorical variables encoded.
    """
    for cat in cat_var_onehot_encoder:
        # Encode the categorical variable using one-hot encoding
        encoded_df = pd.get_dummies(df[cat], drop_first=True, dtype=int)

        # Concatenate the new encoded data with the original DataFrame
        df = pd.concat([df, encoded_df], axis=1)

        # Drop the original column after one-hot encoding
        df.drop(cat, axis=1, inplace=True)

    return df 

def getdummies_encoder(df: pd.DataFrame, cat_var_getdummies_encoder: List[str], threshold: int) -> pd.DataFrame:
    """
    Function to encode categorical variables using get_dummies with thresholding.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_var_getdummies_encoder (List[str]): List of column names for categorical variables.
        threshold (int): Threshold value for rare categories.

    Returns:
        pd.DataFrame: DataFrame with categorical variables encoded using get_dummies.
    """
    for cat in cat_var_getdummies_encoder:
        # Count occurrences of each category
        counts = df[cat].value_counts()
        
        # Identify categories below the threshold
        remaining_cat = counts[counts <= threshold].index
        
        # Replace infrequent categories with 'Uncommon' and apply one-hot encoding
        encoded_df = pd.get_dummies(df[cat].replace(remaining_cat, 'Uncommon'), drop_first=True, dtype=int)
        
        # Concatenate the new encoded data with the original DataFrame
        df = pd.concat([df, encoded_df], axis=1)
        
        # Drop the original column after one-hot encoding
        df.drop(cat, axis=1, inplace=True)
    
    return df

def oversampler(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Function to perform oversampling using SMOTE.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable Series.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Oversampled features and target variable.
    """
    print("Initial distribution of classes in target variable y(Claim):", Counter(y))
    
    # Applying SMOTE for oversampling
    oversampler = SMOTE()
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    
    print("Distribution of classes in target variable y(Claim) after oversampling:", Counter(y_resampled))
    
    return X_resampled, y_resampled

def train_test(X: pd.DataFrame, y: pd.Series, TEST_SIZE: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Function to split the data into training and testing sets.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target variable Series.
        TEST_SIZE (float): Size of the test set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=True, random_state=101)
    return X_train, X_test, y_train, y_test

def feature_selected_kbest(X_train: pd.DataFrame, y_train: pd.Series) -> List[str]:
    """
    Function to select k best features using SelectKBest.

    Args:
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.

    Returns:
        List[str]: List of selected feature names.
    """
    # Initialize SelectKBest
    kbest = SelectKBest(k=10, score_func=f_classif)
    
    # Fit SelectKBest on training data
    kbest.fit(X_train, y_train)
    
    # Get selected feature names
    selected_features = kbest.get_feature_names_out()
    print("\n Feature Selected Kbest : ", selected_features)
    
    return selected_features

def feature_selected_percentile(X_train: pd.DataFrame, y_train: pd.Series) -> List[str]:
    """
    Function to select features based on percentile using SelectPercentile.

    Args:
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.

    Returns:
        List[str]: List of selected feature names.
    """
    # Initialize SelectPercentile
    percentile = SelectPercentile(percentile=20, score_func=f_classif)
    
    # Fit SelectPercentile on training data
    percentile.fit(X_train, y_train)
    
    # Get selected feature names
    selected_features = percentile.get_feature_names_out()
    print("\n Feature Selected Percentile : ", selected_features)
    
    return selected_features

def feature_selected_RFmodel(X_train: pd.DataFrame, y_train: pd.Series) -> List[str]:
    """
    Function to select features using SelectFromModel with RandomForestClassifier.

    Args:
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.

    Returns:
        List[str]: List of selected feature names.
    """
    # Initialize SelectFromModel
    sfm = SelectFromModel(estimator=RandomForestClassifier(), max_features=10, threshold='1.25*mean')
    
    # Fit SelectFromModel on training data
    sfm.fit(X_train, y_train)
    
    # Get selected feature names
    selected_features = sfm.get_feature_names_out()
    print("\n Feature Selected RF Model : ", selected_features)
    
    # Fit RandomForestClassifier to get feature importances
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    
    # Create DataFrame of feature importances
    feat_imps = pd.DataFrame(list(zip(X_train.columns, rf.feature_importances_)),
                             columns=['Feature', 'Importance']).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Plot feature importances
    sns.barplot(x=feat_imps['Importance'], y=feat_imps['Feature'], orient='horizontal', palette='rainbow')
    plt.tight_layout()

    return selected_features
  
def feature_selected_RFE(X_train: pd.DataFrame, y_train: pd.Series) -> List[str]:
    """
    Function to select features using Recursive Feature Elimination (RFE) with RandomForestClassifier.

    Args:
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.

    Returns:
        List[str]: List of selected feature names.
    """
    # Initialize RFE
    rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10, step=4, verbose=2)
    
    # Fit RFE on training data
    rfe.fit(X_train, y_train)
    
    # Get selected feature names
    selected_features = rfe.get_feature_names_out()
    print("\n Feature Selected RFE : ", selected_features)
    
    return selected_features

def feature_selected_Boruta(X_train: pd.DataFrame, y_train: pd.Series) -> List[str]:
    """
    Function to select features using a Boruta-like selection process with RandomForestClassifier.

    Args:
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.

    Returns:
        List[str]: List of selected feature names.
    """
    # Define numpy type aliases to avoid deprecation warnings
    np.int = np.int32
    np.float = np.float64
    np.bool = np.bool_
    
    # Create a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced')

    # Create Boruta feature selector
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, alpha=0.01)

    # Fit Boruta
    boruta_selector.fit(X_train.values, y_train.values)

    # Get selected feature names
    selected_features = X_train.columns[boruta_selector.support_].tolist()

    return selected_features

def feature_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to scale features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Features DataFrame for training.
        X_test (pd.DataFrame): Features DataFrame for testing.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and testing features DataFrames.
    """
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Get feature names
    features = X_train.columns
    
    # Scale training features
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    
    # Scale testing features
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)
    
    return X_train_scaled, X_test_scaled