import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, KFold
import os
from datetime import datetime
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
from typing import List
from sklearn.base import BaseEstimator

def hyperparameter_tuning_LRmodel(LRmodel, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Function to perform hyperparameter tuning for Logistic Regression model.

    Args:
        LRmodel: Logistic Regression model object.
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.

    Returns:
        Any: Best estimator after hyperparameter tuning.
    """
    # Define hyperparameter grid
    param_grid = {'penalty': ['l1','l2','elasticnet'],
              'C': [0.001,0.01,0.1,0.5, 0.8, 0.9],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'multi_class': ['ovr', 'multinomial']
             }

    # Initialize GridSearchCV
    grid_lr = RandomizedSearchCV(LRmodel, param_distributions=param_grid, n_iter=10, cv=2, scoring='roc_auc', verbose=2)

    
    # Perform grid search
    grid_lr.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_lr.best_estimator_
    
    # Print selected hyperparameters
    print("Selected Hyperparameters:")
    print(grid_lr.best_params_)

    return best_model

def hyperparameter_tuning_DTmodel(DTmodel, X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Function to perform hyperparameter tuning for Decision Tree model.

    Args:
        DTmodel: Decision Tree model object.
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.

    Returns:
        Any: Best estimator after hyperparameter tuning.
    """
    # Define hyperparameter grid
    param_grid = param_grid = {'criterion': ['gini','entropy'],
                                'max_features': ['sqrt','log2'],
                                'max_depth': [2,10,50,100,500, 1000]
                                }

    # Initialize GridSearchCV
    grid_dt = GridSearchCV(DTmodel, param_grid=param_grid, verbose=2, cv=5, scoring='roc_auc')
    
    # Perform grid search
    grid_dt.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_dt.best_estimator_

     # Print selected hyperparameters
    print("Selected Hyperparameters:")
    print(grid_dt.best_params_)
    
    return best_model

def hyperparameter_tuning_RFmodel(RFmodel: RandomForestClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Function to perform hyperparameter tuning for Random Forest model.

    Args:
        RFmodel: Random Forest model object.
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.

    Returns:
        RandomForestClassifier: Best estimator after hyperparameter tuning.
    """
    # Define hyperparameter grid
    param_grid = {'n_estimators': [200,400,600,800,1000], 
                  'criterion': ['gini', 'entropy'],
                  'max_features': ['auto','sqrt','log2'],
                  'class_weight': ['balanced', 'balanced_subsample']}
    
    # Initialize GridSearchCV
    grid_rf = RandomizedSearchCV(RFmodel, param_distributions=param_grid, n_iter=10, cv=2, scoring='roc_auc', verbose=2)
    
    # Perform grid search
    grid_rf.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_rf.best_estimator_

    # Print selected hyperparameters
    print("Selected Hyperparameters:")
    print(grid_rf.best_params_)
    
    return best_model

def hyperparameter_tuning_XGBmodel(XGBmodel, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Function to perform hyperparameter tuning for XGBoost model.

    Args:
        XGBmodel: XGBoost model object.
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.

    Returns:
        XGBClassifier: Best estimator after hyperparameter tuning.
    """
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'min_child_weight': [1, 2, 3, 4],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    # Initialize GridSearchCV
    grid_xgb = RandomizedSearchCV(XGBmodel, param_distributions=param_grid, n_iter=10, cv=2, scoring='roc_auc', verbose=2)
    
    # Perform grid search
    grid_xgb.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_xgb.best_estimator_
    
    # Print selected hyperparameters
    print("Selected Hyperparameters:")
    print(grid_xgb.best_params_)

    return best_model

def save_results(model_perfs: pd.DataFrame) -> None:
    """
    Function to save model performances to a CSV file.

    Args:
        model_perfs (pd.DataFrame): DataFrame containing model performances.

    Returns:
        None
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define folder to save results
    results_folder = "data/processed_data"
    
    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    
    # Define path for saving results
    results_path = os.path.join(results_folder, f"model_performances_{timestamp}.csv")
    
    # Save model performances to CSV file
    model_perfs.to_csv(results_path, index=False)
    
    print("Results saved in /results")

def perform_cross_validation(model, X_train: pd.DataFrame, y_train: pd.Series, cv=5, scoring='accuracy') -> np.ndarray:
    """
    Function to perform cross-validation for a given model.

    Args:
        model: Model object.
        X_train (pd.DataFrame): Features DataFrame for training.
        y_train (pd.Series): Target variable Series for training.
        cv (int): Number of folds for cross-validation.
        scoring (str): Scoring metric for evaluation.

    Returns:
        np.array: Array of cross-validation scores.
    """
    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    mean_score = np.mean(scores)
    std_deviation = np.std(scores)
    
    # Print model name and cross-validation scores
    print(f"\nModelo: {type(model).__name__}")
    print("Scores de validación cruzada:", scores)
    print(f"Mean score: {mean_score}")
    print(f"Standard deviation: {std_deviation}")
    
    return scores

def plot_decision_tree(DTmodel: BaseEstimator, 
                       feature_names: List[str], 
                       class_names: List[str], 
                       max_depth: int = None) -> None:
    """
    Function to plot a decision tree.

    Args:
        DTmodel: Decision Tree model object.
        feature_names (list): List of feature names.
        class_names (list): List of class names.
        max_depth (int): Maximum depth of the tree to plot.

    Returns:
        None
    """
    plt.figure(figsize=(20, 10))  
    plot_tree(DTmodel, feature_names=feature_names, class_names=class_names, filled=True, max_depth=max_depth)
    plt.show()

    tree_rules = export_text(DTmodel, feature_names=list(feature_names))
    print(tree_rules)