import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,ConfusionMatrixDisplay, PrecisionRecallDisplay
import os
import pickle
import joblib
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
import seaborn as sns

def save_model(model: BaseEstimator, model_name: str) -> None:
    """
    Function to save a trained model.

    Args:
        model (BaseEstimator): Trained model object.
        model_name (str): Name of the model.

    Returns:
        None
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define folder to save model
    model_folder = os.path.join("models", model_name)
    
    # Create model folder if it doesn't exist
    os.makedirs(model_folder, exist_ok=True)
    
    # Define path for saving model
    model_path = os.path.join(model_folder, f"{model_name}_{timestamp}.pkl")
    
    # Save model to file
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print("Models saved in /models")
    
def load_best_model() -> BaseEstimator:
    """
    Function to load the best trained model from the 'best_model' folder.

    Returns:
        BaseEstimator: Loaded model object.
    """
    # Define folder path
    folder_path = os.path.join("models", "best_model") 
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter pickle files
    pkl_files = [file for file in files if file.endswith(".pkl")]
    
    # Raise error if no pickle files found
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the models folder.")
    
    # Get the latest modified pickle file
    latest_file = max(pkl_files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    
    # Get the path of the latest model file
    model_path = os.path.join(folder_path, latest_file)
    
    # Load the model from the file
    model = joblib.load(model_path)
    
    return model

def train_and_evaluate_model(model: BaseEstimator, 
                             X_train: pd.DataFrame, 
                             X_test: pd.DataFrame, 
                             y_train: pd.Series, 
                             y_test: pd.Series, 
                             models: List[BaseEstimator], 
                             model_names: List[str], 
                             accuracy_scores: List[float], 
                             precision_scores: List[float], 
                             recall_scores: List[float], 
                             f1_scores: List[float], 
                             roc_auc_scores: List[float]) -> Tuple[List[str], List[float], List[float], List[float], List[float], List[float]]:
    """
    Function to train and evaluate a model, and save the model.

    Args:
        model (BaseEstimator): Model object.
        X_train (pd.DataFrame): Features DataFrame for training.
        X_test (pd.DataFrame): Features DataFrame for testing.
        y_train (pd.Series): Target variable Series for training.
        y_test (pd.Series): Target variable Series for testing.
        models (List[BaseEstimator]): List to store trained models.
        model_names (List[str]): List to store names of trained models.
        accuracy_scores (List[float]): List to store accuracy scores of models.
        precision_scores (List[float]): List to store precision scores of models.
        recall_scores (List[float]): List to store recall scores of models.
        f1_scores (List[float]): List to store F1 scores of models.
        roc_auc_scores (List[float]): List to store ROC AUC scores of models.

    Returns:
        Tuple[List[str], List[float], List[float], List[float], List[float], List[float]]: Lists containing model names and evaluation scores.
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    # Print evaluation scores
    print("Accuracy Score:", accuracy)
    print("Precision Score:", precision)
    print("Recall Score:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)

    # Crear una figura con tres subplots en una fila
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot matriz de confusión
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0])
    axes[0].set_title("Matriz de Confusión")

    # Plot curva de precisión-recall
    pr_display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=axes[1])
    axes[1].set_title("Curva de Precisión-Recall")

    # Plot curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    axes[2].plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
    axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('Tasa de Falsos Positivos')
    axes[2].set_ylabel('Tasa de Verdaderos Positivos')
    axes[2].set_title('Curva ROC')
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    # Save the model (assuming a function save_model exists)
    save_model(model, type(model).__name__)

    # Append model and evaluation scores to respective lists
    models.append(model)
    model_names.append(str(model))
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)

    return model_names, accuracy_scores, precision_scores, recall_scores, f1_scores, roc_auc_scores

def calculate_odds_ratio(logistic_model: LogisticRegression, feature_names: list) -> Dict[str, float]:
    """
    Function to calculate odds ratio for logistic regression model coefficients.

    Args:
        logistic_model (LogisticRegression): Trained logistic regression model.
        feature_names (list): List of feature names.

    Returns:
        dict: Dictionary containing feature names as keys and their respective odds ratios as values.
    """
    # Get the coefficients of the model
    coef = logistic_model.coef_
    
    # Apply exponential function to coefficients to get the odds ratios
    odds_ratio = np.exp(coef)
    
    # Create a dictionary to store odds ratios along with feature names
    odds_ratio_dict = {}
    for i, feature_name in enumerate(feature_names):
        odds_ratio_dict[feature_name] = odds_ratio[0][i]
    
    return odds_ratio_dict

def calculate_feature_importance(model: BaseEstimator, X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate feature importance for a given model.

    Args:
        model (BaseEstimator): Trained model object with a `feature_importances_` attribute.
        X_train (pd.DataFrame): DataFrame containing the features used for training.

    Returns:
        pd.DataFrame: DataFrame containing feature importance values and cumulative percentage.
    """
    # Get feature importance from the model
    feature_importance = model.feature_importances_
    
    # Create a DataFrame to hold feature importance values
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
    
    # Sort the DataFrame by importance values in descending order
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    # Calculate cumulative percentage of importance
    importance_df['Cumulative Percentage'] = importance_df['Importance'].cumsum()
    
    return importance_df


