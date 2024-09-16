import os
import mlflow
import pandas as pd
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
import matplotlib.pyplot as plt

def get_random_themes(view_name, limit=10, use_global=False):
    """
    Generates and saves random themes from the given Spark SQL view or table.

    Parameters:
    view_name (str): The name of the Spark SQL view or table containing the 'Theme' column.
    limit (int): Number of random themes to select. Default is 10.
    use_global (bool): Whether to use the globally saved themes if they exist. Default is True.

    Returns:
    list: A list of random themes.
    """
    global saved_themes  # Declare the variable as global to update it

    if use_global and 'saved_themes' in globals():
        # If use_global is True and saved_themes exists in the global scope, use the global themes
        print("Using previously saved themes:", saved_themes)
    else:
        # Generate and save random themes
        random_themes_query = f"""
        SELECT Theme
        FROM {view_name}
        ORDER BY RAND()
        LIMIT {limit}
        """
        
        # Execute the query
        random_themes_df = spark.sql(random_themes_query)
        
        # Extract themes into a list
        saved_themes = [row['Theme'] for row in random_themes_df.collect()]
        
        print("Random themes saved:", saved_themes)
    
    return saved_themes


def plot_ndcg_scores(ndcg_view):
    """
    Plots NDCG scores for the saved themes.

    Parameters:
    ndcg_view (str): The name of the Spark SQL view or table containing NDCG data.
    
    Returns:
    None
    """
    # SQL query to get NDCG values for the saved themes
    theme_query = f"""
    SELECT Theme, `NDCG@10`, `NDCG@20`, `NDCG@30`, `NDCG@40`, `NDCG@50`, `NDCG@100`
    FROM {ndcg_view}
    WHERE Theme IN ({','.join(f"'{theme}'" for theme in saved_themes)})
    """

    # Execute the query and create a DataFrame
    theme_data_df = spark.sql(theme_query)
    theme_data_pandas_df = theme_data_df.toPandas()

    # Filter out themes with incomplete NDCG data
    valid_themes = theme_data_pandas_df.dropna(subset=['NDCG@10', 'NDCG@20', 'NDCG@30', 'NDCG@40', 'NDCG@50', 'NDCG@100'])['Theme'].unique()

    # Plot
    plt.figure(figsize=(12, 8))
    for theme in saved_themes:
        if theme in valid_themes:
            theme_df = theme_data_pandas_df[theme_data_pandas_df['Theme'] == theme]
            plt.plot(
                theme_df.columns[1:],  # x-axis (NDCG columns)
                theme_df.iloc[0, 1:],  # y-axis (scores)
                marker='o',
                label=theme
            )
        else:
            print(f"Theme '{theme}' does not have complete data and is excluded from the plot.")

    # Customize the plot
    plt.xlabel('NDCG')
    plt.ylabel('Score')
    plt.title('NDCG Scores for Selected Themes')
    plt.ylim(0, 1)
    plt.legend(title='Themes', bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_lowest_ndcg_for_themes_spark(df):
    """
    Plots NDCG scores for different themes over various NDCG thresholds, starting from a Spark DataFrame.

    Parameters:
    spark_df (pyspark.sql.DataFrame): Spark DataFrame containing themes and NDCG scores at different thresholds.

    Returns:
    None
    """
    
    # List of NDCG columns
    ndcg_columns = ['NDCG@10', 'NDCG@20', 'NDCG@30', 'NDCG@40', 'NDCG@50', 'NDCG@100']

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot a line for each theme
    for index, row in df.iterrows():
        theme = row['Theme']
        plt.plot(ndcg_columns, row[ndcg_columns], marker='o', label=theme)

    # Customize the plot
    plt.xlabel('NDCG Thresholds')
    plt.ylabel('NDCG Scores')
    plt.title('NDCG Scores for Different Themes Across Thresholds')
    plt.ylim(0, 1)
    plt.legend(title='Themes', bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
    plt.grid(True)
    plt.tight_layout()

    # Set y-axis to always range from 0 to 1
    plt.ylim(0, 1)

    # Show the plot
    plt.show()

def plot_ndcg_line_graph(df):
    """
    Plots a line graph of Overall NDCG@k values.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'k' and 'Overall NDCG@k' columns.

    Returns:
    None
    """

    if 'k' not in df.columns or 'Overall NDCG@k' not in df.columns:
        raise ValueError("DataFrame must contain 'k' and 'Overall NDCG@k' columns.")

    plt.figure(figsize=(12, 6))
    plt.plot(df['k'], df['Overall NDCG@k'], marker='o', linestyle='-', color='skyblue', linewidth=2, markersize=8)

    plt.xlabel('k')
    plt.ylabel('Overall NDCG@k')
    plt.title('Overall NDCG@k for Different k Values')
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_lowest_ndcg_bar_graph(df):
    """
    Plots a bar graph of average NDCG values for different themes.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Theme' and 'Average NDCG' columns.

    Returns:
    None
    """
    # Ensure the DataFrame has the correct columns
    if 'Theme' not in df.columns or 'Overall Average NDCG' not in df.columns:
        raise ValueError("DataFrame must contain 'Theme' and 'Average NDCG' columns.")

    # Set the size of the plot
    plt.figure(figsize=(12, 8))

    # Create the bar plot
    plt.bar(df['Theme'], df['Overall Average NDCG'], color='skyblue')

    # Rotate x-axis labels for better radability
    plt.xticks(rotation=45, ha='right')

    # Customize the plot
    plt.xlabel('Theme')
    plt.ylabel('Overall Average NDCG')
    plt.title('Lowest Overall Average NDCG Scores by Theme')
    plt.ylim(0, 0.6)
    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_lowest_ndcg_for_themes_subplots(pandas_df, ax, ndcg_value):
    """
    Plots NDCG scores for different themes over various NDCG thresholds on a specific axis.

    Parameters:
    df (pyspark.sql.DataFrame): Spark DataFrame containing themes and NDCG scores at different thresholds.
    ax (matplotlib.axes.Axes): The axis on which to plot the data.
    ndcg_value (int): The NDCG threshold value (e.g., 10, 20, etc.) to include in the subplot title.

    Returns:
    None
    """
    
    # List of NDCG columns
    ndcg_columns = ['NDCG@10', 'NDCG@20', 'NDCG@30', 'NDCG@40', 'NDCG@50', 'NDCG@100']

    # Plot a line for each theme on the given axis
    for index, row in pandas_df.iterrows():
        theme = row['Theme']
        ax.plot(ndcg_columns, row[ndcg_columns], marker='o', label=theme)

    # Customize the subplot
    ax.set_xlabel('NDCG Thresholds', fontsize=10)
    ax.set_ylabel('NDCG Scores', fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'NDCG Scores for Different Themes (NDCG@{ndcg_value})', fontsize=12)  # Adding the NDCG threshold to the title
    ax.grid(True)
    
    # Adjust legend placement and size
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)

def plot_six_ndcg_subplots(dfs, ndcg_values):
    """
    Creates a 3x2 grid of subplots and calls the plot_lowest_ndcg_for_themes_subplots function on each.

    Parameters:
    dfs (list of pyspark.sql.DataFrame): List of 6 Spark DataFrames to plot.
    ndcg_values (list of int): List of NDCG threshold values corresponding to each DataFrame (e.g., [10, 20, 30, 40, 50, 100]).
    
    Returns:
    None
    """
    
    # Create a 3x2 grid of subplots with an increased figure size
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))  # Changed to 3 rows, 2 columns
    
    # Flatten the axes array to make iteration easier
    axes = axes.flatten()
    
    # Plot each DataFrame in its own subplot, adding the respective NDCG value to the title
    for i, df in enumerate(dfs):
        plot_lowest_ndcg_for_themes_subplots(df, axes[i], ndcg_values[i])
    
    # Adjust the layout to prevent overlap
    plt.tight_layout(pad=3.0)
    
    # Show the final plot with all subplots
    plt.show()

def download_and_load_csvs_for_model(run_id: str, artifact_path: str, model: str) -> dict:
    """
    Downloads artifacts from MLflow, specifically targets the model's subfolder, and loads all CSV files into a dictionary of DataFrames.

    Parameters:
    - run_id (str): The MLflow run ID.
    - artifact_path (str): The base path within the artifact directory.
    - model (str): The specific model subfolder to look into.

    Returns:
    - dict: A dictionary where the keys are file paths and the values are DataFrames.
    """
    # Download the entire artifact directory to a local temporary directory
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    
    # Initialize a dictionary to store DataFrames with their file paths as keys
    dataframes_dict = {}
    
    # Construct the full model path to look for
    model_path = os.path.join(local_dir, model)
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified model path does not exist: {model_path}")
    
    # Traverse the model directory to find all CSV files
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith(".csv"):
                # Construct the full file path
                file_path = os.path.join(root, file)
                print(f"Loading {file_path}")
                
                # Load the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Store the DataFrame in the dictionary with the file path as key
                dataframes_dict[file_path] = df
    
    return dataframes_dict

def load_and_convert_artifacts_to_spark(run_id, artifact_path):
    # Download the artifacts from MLflow
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    
    # Load CSV files into Pandas DataFrames
    pandas_ndcg_table = pd.read_csv(local_path + "/ndcg_table.csv")
    overall_ndcg_table = pd.read_csv(local_path + "/overall_ndcg_table.csv")
    lowest_ndcg_spec_col_10 = pd.read_csv(local_path + "/lowest_ndcg_spec_col_10.csv")
    lowest_ndcg_spec_col_20 = pd.read_csv(local_path + "/lowest_ndcg_spec_col_20.csv")
    lowest_ndcg_spec_col_30 = pd.read_csv(local_path + "/lowest_ndcg_spec_col_30.csv")
    lowest_ndcg_spec_col_40 = pd.read_csv(local_path + "/lowest_ndcg_spec_col_40.csv")
    lowest_ndcg_spec_col_50 = pd.read_csv(local_path + "/lowest_ndcg_spec_col_50.csv")
    lowest_ndcg_spec_col_100 = pd.read_csv(local_path + "/lowest_ndcg_spec_col_100.csv")
    lowest_averg_ndcg_per_col = pd.read_csv(local_path + "/lowest_averg_ndcg_per_col.csv")
    df_merged = pd.read_csv(local_path + "/df_merged.csv")
    
    # Combine specific column DataFrames into a list
    dfs = [lowest_ndcg_spec_col_10, lowest_ndcg_spec_col_20, lowest_ndcg_spec_col_30, 
           lowest_ndcg_spec_col_40, lowest_ndcg_spec_col_50, lowest_ndcg_spec_col_100]
    
    # Convert Pandas DataFrames to Spark DataFrames
    spark_ndcg = spark.createDataFrame(pandas_ndcg_table)
    spark_lowest_averg_ndcg_per_col = spark.createDataFrame(lowest_averg_ndcg_per_col)
    
    # Register the Spark DataFrames as SQL Views
    spark_ndcg.createOrReplaceTempView("ndcg_view")
    spark_lowest_averg_ndcg_per_col.createOrReplaceTempView("lowest_averg_ndcg_per_col")
    
    return pandas_ndcg_table, spark_ndcg, spark_lowest_averg_ndcg_per_col, dfs, overall_ndcg_table


def melt_and_combine_dfs(df1, df2, id_vars, value_vars, source_labels):
    """
    Melt and combine two DataFrames.

    Parameters:
    - df1 (pd.DataFrame): The first DataFrame to melt.
    - df2 (pd.DataFrame): The second DataFrame to melt.
    - id_vars (list): List of columns to use as identifier variables.
    - value_vars (list): List of columns to unpivot.
    - source_labels (tuple): Tuple with labels for the source DataFrames (e.g., ('DF1', 'DF2')).

    Returns:
    - pd.DataFrame: The combined melted DataFrame.
    """
    
    # Melt the first DataFrame
    df1_melted = df1.melt(id_vars=id_vars, value_vars=value_vars, var_name='Metric', value_name='Value')
    df1_melted['Source'] = source_labels[0]
    
    # Melt the second DataFrame
    df2_melted = df2.melt(id_vars=id_vars, value_vars=value_vars, var_name='Metric', value_name='Value')
    df2_melted['Source'] = source_labels[1]
    
    # Combine DataFrames
    df_combined = pd.concat([df1_melted, df2_melted])
    
    return df_combined

def extract_model_name(path):
    # Remove the base directory part '/dbfs/FileStore/'
    return path.replace('/dbfs/FileStore/', '').replace('/', '_')