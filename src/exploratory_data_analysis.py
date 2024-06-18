import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import probplot
from scipy.stats import chi2_contingency
import scipy.stats as stats
from scipy.stats import shapiro

def categorical_analysis(df: pd.DataFrame, cat_var_names: list) -> pd.DataFrame:
    """
    Analyzes categorical variables in the DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame to analyze.
        cat_var_names (list): List of categorical variable names.

    Returns:
        pd.DataFrame: DataFrame containing feature frequencies.
    """
    print("\n--- Categorical Analysis ---")
    cat_df = pd.DataFrame(columns=['Feature', 'Frequency'])

    for col in cat_var_names:
        # Frequency graph for top 20 categories
        print(f"\n--- {col} - Top Categories ---")
        print("Frequency:")
        value_counts = df[col].value_counts()
        top_20 = value_counts.head(20)

        # Convert to DataFrame and calculate percentage of participation
        top_20_df = pd.DataFrame(top_20).reset_index()
        top_20_df.columns = ['Category', 'Frequency']
        total_count = value_counts.sum()
        top_20_df['Percentage'] = (top_20_df['Frequency'] / total_count) * 100
        top_20_df['Cumulative Percentage'] = top_20_df['Percentage'].cumsum()  # Calculate cumulative percentage
        print("\nTop Categories with Percentage of Participation:")
        print(top_20_df)

        # Plot both bar and pie charts side by side
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot for top 20 categories
        axes[0].bar(top_20.index, top_20.values)
        axes[0].set_title(f"Top 20 {col} Categories")
        axes[0].set_ylabel('Frequency')
        axes[0].tick_params(axis='x', rotation=90)

        # Pie chart for top 5 categories
        top_5 = top_20_df.head(5)
        axes[1].pie(top_5['Frequency'], labels=top_5['Category'], autopct='%1.1f%%', startangle=140)
        axes[1].set_title(f"Top 5 {col} Categories")
        axes[1].axis('equal')

        plt.show()

        # Features Frequency Table
        new_row = pd.DataFrame({
            'Feature': [col],
            'Frequency': [df[col].nunique()]
        })
        cat_df = pd.concat([cat_df, new_row], ignore_index=True)

    print("\nFeatures Frequency Table:\n", cat_df)

def numeric_analysis(df: pd.DataFrame, num_var_names: list) -> None:
    """
    Analyzes numeric variables in the DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame to analyze.
        num_var_names (list): List of numeric variable names.

    Returns:
        None
    """
    print("\n--- Numeric Analysis ---")
    for col in num_var_names:
        print(f"\n--- {col} ---")
        print("Basic Statistics:")
        print(df[col].describe())

        # Create the subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Distribution plot of the numeric variable
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"{col} Distribution")

        # Plot 2: Boxplot of the numeric variable
        sns.boxplot(data=df, y=col, ax=axes[1])
        axes[1].set_title(f"{col} Boxplot")

        # Plot 3: Q-Q plot of the numeric variable
        probplot(df[col], plot=axes[2], rvalue=True, dist='norm')
        axes[2].set_title(f"{col} Q-Q Plot")

        # Shapiro-Wilk test to determine normality of distribution
        stat, p = shapiro(df[col])
        if p > 0.05:
            result = "normally distributed"
        else:
            result = "not normally distributed"

        # Print the result of Shapiro-Wilk test
        print(f"Shapiro-Wilk Test for {col}: p-value = {p:.4f}, Result = {result}")

        # Adjust layout and display the figure
        plt.tight_layout()
        plt.show()

def correlation_analysis(df: pd.DataFrame, num_var_names: list) -> None:
    """
    Performs correlation analysis on numeric variables in the DataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame to analyze.
        num_var_names (list): List of numeric variable names.

    Returns:
        None
    """
    print("\n--- Correlation Analysis ---")
    corr_matrix = df[num_var_names].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def target_relationship(df: pd.DataFrame, cat_var_names: list, num_var_names: list, num_var_df: pd.DataFrame, y_df: pd.DataFrame, y_name: str) -> None:
    """
    Analyzes the relationship between predictors and the target variable.

    Args:
        df (pd.DataFrame): The pandas DataFrame to analyze.
        cat_var_names (list): List of categorical variable names.
        num_var_names (list): List of numeric variable names.
        num_var_df (pd.DataFrame): DataFrame of numeric variables.
        y_df (pd.DataFrame): Target DataFrame.
        y_name (str): Name of the target variable.

    Returns:
        None
    """
    print("\n--- Target Relationship ---")
    # Categorical Variables
    for col in cat_var_names:
        print(f"\n--- {col} vs {y_name} ---")
        # Create contingency table between the categorical variable and the target variable
        contingency_table = pd.crosstab(df[col], df[y_name])
        # Perform chi-square test
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"Chi-square Test for {col} vs {y_name}:")
        print(f"Chi-square Statistic: {chi2}")
        print(f"P-Value: {p}")
        if p < 0.05:
            print("There is a significant relationship between the variables.")
        else:
            print("There is no significant relationship between the variables.")
    # Numeric variables
    for col in num_var_names:
        print(f"\n--- {col} vs {y_name} ---")
        group1 = df[df[y_name] == 'Yes'][col]
        group2 = df[df[y_name] == 'No'][col]
        t_stat, p_value = stats.ttest_ind(group1, group2)
        print(f"T-test for {col} vs {y_name}:")
        print(f"T-Statistic: {t_stat}")
        print(f"P-Value: {p_value}")

    print("\n--- Pair Plot Claim ---:")
    pairplot_df = pd.concat([num_var_df, y_df], axis=1)
    sns.pairplot(pairplot_df, hue=y_name, palette='coolwarm')
    plt.tight_layout()

def plot_horizontal_bar(df: pd.DataFrame, 
                         cat_var_name: str, 
                         num_var_name: str, 
                         top_n: int = 10) -> None:
    """
    Generate a horizontal bar plot from a DataFrame using specified parameters.

    Parameters:
        df (pd.DataFrame): The pandas DataFrame.
        cat_var_name (str): The name of the categorical column.
        num_var_name (str): The name of the numerical column.
        top_n (int, optional): Number of top categories to display (default is 10).
        
    Returns:
        None
    """
    # Filter rows with positive values in numeric column
    filtered_df = df[df[num_var_name] > 0]
    
    # Group by categorical column and calculate sum and mean of numeric column
    grouped_df = filtered_df.groupby([cat_var_name]).agg({num_var_name: ['sum', 'mean']})
    
    # Sort values by sum
    sorted_df = grouped_df.sort_values(by=(num_var_name, 'sum'), ascending=False)
    
    # Select top n rows
    top_n_df = sorted_df.head(top_n)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot sum of categories
    ax1.barh(top_n_df.index, top_n_df[(num_var_name, 'sum')], color='b', label='Sum')
    ax1.set_xlabel(num_var_name.capitalize() + ' Sum')
    ax1.set_ylabel(cat_var_name.capitalize())
    ax1.set_title(f'Top {top_n} {cat_var_name.capitalize()} by Sum {num_var_name.capitalize()}')
    
    # Plot mean of categories
    ax2.barh(top_n_df.index, top_n_df[(num_var_name, 'mean')], color='g', label='Mean')
    ax2.set_xlabel(num_var_name.capitalize() + ' Mean')
    ax2.set_title(f'Top {top_n} {cat_var_name.capitalize()} by Mean {num_var_name.capitalize()}')
    
    # Show legend
    ax1.legend()
    ax2.legend()
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def plot_bar_target_variable(df: pd.DataFrame, 
                             obj_var: str, 
                             feature: str, 
                             top_n: int = 10) -> None:
    """
    Generate a bar plot showing the distribution of target variable (0 or 1) with respect to another variable.

    Parameters:
        df (pd.DataFrame): The pandas DataFrame.
        obj_var (str): The name of the target variable column (0 or 1).
        feature (str): The name of the categorical variable column.
        top_n (int, optional): Number of top subcategories to display. Default is 10.
        
    Returns:
        None
    """
    # Group by the feature column and count the number of records for each category
    counts = df[feature].value_counts()

    # Filter categories by the top n counts
    top_n_categories = counts.head(top_n).index.tolist()

    # Filter DataFrame by the top n categories
    filtered_df = df[df[feature].isin(top_n_categories)]

    # Group by the feature column and count the number of records for each combination of feature and obj_var
    grouped_df = filtered_df.groupby([feature, obj_var]).size().unstack()

    # Plot bar chart
    grouped_df.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.title(f'Top {top_n} {feature.capitalize()} by {obj_var.capitalize()}')
    plt.xticks(rotation=90)
    plt.legend(title=obj_var.capitalize())
    plt.show()

def plot_histograms_seaborn(df: pd.DataFrame, num_var_name: str, y_name: str):
    """
    Plot separate histograms of a numerical variable for each binary target variable category using seaborn.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        num_var_name (str): Name of the numerical variable.
        y_name (str): Name of the target variable (assumed binary).
    """
    # Separate the data based on the target variable
    no_claim = df[df[y_name] == 'No']
    yes_claim = df[df[y_name] == 'Yes']

    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    # Plot 1: Distribution plot of no claim
    sns.histplot(no_claim[num_var_name], kde=True, color='blue', ax=axes[0])
    axes[0].axvline(no_claim[num_var_name].mean(), color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {no_claim[num_var_name].mean():.2f}')
    axes[0].set_title(f"Histogram of {num_var_name} for No Claim")
    axes[0].set_xlabel(num_var_name)
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Plot 2: Distribution plot of claim
    sns.histplot(yes_claim[num_var_name], kde=True, color='blue', ax=axes[1])
    axes[1].axvline(yes_claim[num_var_name].mean(), color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {yes_claim[num_var_name].mean():.2f}')
    axes[1].set_title(f"Histogram of {num_var_name} for Yes Claim")
    axes[1].set_xlabel(num_var_name)
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()

def plot_categorical_comparison(df: pd.DataFrame, cat_var_name: str, cat_var_name2: str):
    """
    Plot a comparison of categorical variables.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cat_var_name (str): Name of the first categorical variable.
        cat_var_name2 (str): Name of the second categorical variable.
    """
    # Get unique categories of cat_var_name
    unique_categories = df[cat_var_name].unique()
    
    # Calculate the number of subplots needed
    num_subplots = len(unique_categories)
    
    # Create subplots
    fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5), sharey=True)
    
    # Iterate over unique categories and create a bar plot for each one
    for i, cat in enumerate(unique_categories):
        ax = axes[i] if num_subplots > 1 else axes  # If only one subplot, use the same axis
        x_counts = df[df[cat_var_name] == cat][cat_var_name2].value_counts()
        x_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f"{cat_var_name}: {cat}")
        ax.set_xlabel(cat_var_name2)
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


