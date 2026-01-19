import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import pickle  # optional, for saving μ and σ


def remove_negative_mic(df):
    """
    Remove negative MIC value from the dataframe .
    Requires column named 'value' in the dataframe.
    """
    total_len = len(df)
    df = df.copy()
    df = df[df["value"] > 0]

    new_len = len(df)

    negatives_count = total_len - new_len

    print(
        f"Negative remove operation: Intial Len: {total_len} Neg Count: {negatives_count} New length: {len(df)} "
    )

    return df, negatives_count


def get_unique_bacteria(df):
    """
    Return a list of unique values from the 'bacterium' column.
    """
    return df["bacterium"].dropna().unique().tolist()


def get_bacteria_by_type(df, bacterium="E. coli"):
    """
    Return all rows in the dataframe corresponding to the specified bacterium.
    """
    # Ensure no leading/trailing spaces in bacterium column
    df = df.copy()
    df["bacterium"] = df["bacterium"].str.strip()  # rmove trailing whitespace

    # Filter rows
    filtered_df = df[df["bacterium"] == bacterium]

    print(f"Total records for E. coli is : {len(filtered_df)}")

    return filtered_df


def build_e_coli_db(grampa_csv_path="data/grampa.csv", save_path="./ecoli.csv"):
    df = pd.read_csv(grampa_csv_path)

    print(f"Original DB lenght : {len(df)}")
    df = get_bacteria_by_type(df, bacterium="E. coli")

    df = df.drop(df.columns[0], axis=1)  # Remove unamed first column (index)

    df.to_csv(save_path, index=False)
    print(f"E Coli Data saved successfully to File: {save_path}")


def count_and_remove_duplicates(df):
    """
    Count duplicate rows and return:
      - cleaned_df: duplicates removed, keeping one entry per sequence
      - duplicate_count: how many duplicate rows existed

    Behavior:
      - Identifies duplicates based on the 'sequence' column (case-insensitive, ignores leading/trailing spaces)
      - For duplicate sequences, averages the 'value' column
      - Keeps only one row per sequence in the cleaned dataframe
      - Prints info about duplicate removal and group counts
    """
    df = df.copy()

    # Normalize sequence: strip spaces and convert to lowercase for consistent comparison
    df["sequence_normalized"] = df["sequence"].str.strip().str.lower()

    # Identify duplicate rows using normalized sequence
    # keep=False marks all duplicates as True
    duplicate_mask = df.duplicated(subset=["sequence_normalized"], keep=False)
    duplicate_count = duplicate_mask.sum()

    # Count total duplicate groups (case-insensitive & stripped)
    duplicate_groups_count = (df["sequence_normalized"].value_counts() > 1).sum()

    if duplicate_count > 0:
        # For duplicates, compute mean 'value' per normalized sequence
        averaged_values = (
            df.groupby("sequence_normalized")["value"].mean().reset_index()
        )
        print("Averaged value structure:\n", averaged_values.head())

        # Drop duplicate rows based on normalized sequence, keep first occurrence
        cleaned_df = df.drop_duplicates(subset=["sequence_normalized"], keep="first")

        # Update 'value' column with the averaged values for duplicates
        cleaned_df = cleaned_df.drop(columns=["value"]).merge(
            averaged_values, on="sequence_normalized", how="left"
        )

    else:
        cleaned_df = df

    print(f"Duplicate Remove Operation: Total duplicate rows: {duplicate_count}")
    print(f"Total duplicated groups found: {duplicate_groups_count}")
    print(f"Original DF length: {len(df)} | Cleaned DF length: {len(cleaned_df)}")

    # Drop helper column before returning
    cleaned_df = cleaned_df.drop(columns=["sequence_normalized"])

    return cleaned_df, duplicate_count


def normalize_mic(df):
    """
    Normalize the 'value' column in the dataframe to the range 0–1.
    """
    min_val = df["value"].min()
    max_val = df["value"].max()

    df["normalized_value"] = (df["value"] - min_val) / (max_val - min_val)
    return df

def normalize_log_mic(df, value_col="value", save_stats_path=None, stats=None):
    """
    Normalize MIC values with log10 transform and z-score normalization.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing MIC values in `value_col`.
    value_col : str
        Column name of MIC values (default 'value').
    save_stats_path : str or None
        Optional path to save mean and std for inference.
    stats : dict or None
        If provided, should contain 'mean' and 'std' for normalization (inference mode).

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with a new column 'normalized_value'.
    stats : dict
        Dictionary containing 'mean' and 'std' of the training data (if stats not provided).
    """

    # Step 1: Log-transform the MIC
    df["log_mic"] = np.log10(df[value_col])

    # Step 2: Compute or use provided mean and std
    if stats is None:
        mean_val = df["log_mic"].mean()
        std_val = df["log_mic"].std()
        stats = {"mean": mean_val, "std": std_val}
    else:
        mean_val = stats["mean"]
        std_val = stats["std"]

    # Step 3: Z-score normalization
    df["normalized_value"] = (df["log_mic"] - mean_val) / std_val

    #added new comment
    # Step 4: Optionally save mean and std for inference
    if save_stats_path is not None and stats is not None:
        with open(save_stats_path, "wb") as f:
            pickle.dump(stats, f)

    return df, stats


def count_records_by_db(df):
    """
    Count rows grouped by the 'database' column and sort descending.
    """
    counts = df["database"].value_counts().sort_values(ascending=False)
    print(f"Counts by database is : {counts}")
    return counts


def save_preprocessed_ecoli(data_root=None, grampa_file=None):
    grampa_path = f"{data_root}/{grampa_file}"
    ecoli_path = f"{data_root}/ecoli_original.csv"
    ecoli_normalized_path = f"{data_root}/ecoli_mic_normalized.csv"

    build_e_coli_db(grampa_path, save_path=ecoli_path)

    df = pd.read_csv(ecoli_path)

    df, negatives_count = remove_negative_mic(df)
    df, duplicate_count = count_and_remove_duplicates(df)

    # df, train_stats = normalize_log_mic(df, value_col="value", save_stats_path="mic_stats.pkl")


    count_records_by_db(df)  # Count How many records belongs to each db

    df.to_csv(ecoli_normalized_path, index=False)
    print(f"Normalized Ecoli CSV saved to : {ecoli_normalized_path} successfully.")
    return df


def get_unique_column_values(df, column_name):
    unique_values = df[column_name].unique()
    print(f"Unique Db used are: {unique_values}")
    return unique_values

def build_sequence_group_by_length(csv_file)-> dict:
    """
    Groups sequences by length from a DataFrame with a 'sequence' column.

    Groups:
    - group_1_5     : length 1–5
    - group_6_25    : length 6–25
    - group_26_49   : length 26–49
    - group_50_plus : length ≥50

    Returns
    -------
    dict
    Dictionary mapping each group name to a list of sequences.
    """
   
    df = pd.read_csv(csv_file)

    # Initialize groups as dictionaries with lists
    groups = {
        'group_1_5': [],  # length < 6
        'group_6_25': [],  # 6 <= length < 26
        'group_26_49': [],  # 26 <= length < 50
        'group_50_plus': []   # length >= 50
    }

    # Iterate over sequences and assign to groups
    for seq in df['sequence']:
        l = len(seq)
        if l < 6:
            groups['group_1_5'].append(seq)
        elif 6 <= l < 26:
            groups['group_6_25'].append(seq)
        elif 26 <= l < 50:
            groups['group_26_49'].append(seq)
        else:  
            groups['group_50_plus'].append(seq)

    # Check results
    total_values = 0
    for _, v in groups.items():
        total_values += len(v)
    
    assert total_values == len(df)

    print(groups.keys())

    return groups



if __name__ == "__main__":
    df = save_preprocessed_ecoli(data_root="data", grampa_file="grampa.csv")
    print(f"Ecoli Database Successfully build from Grampa.")
