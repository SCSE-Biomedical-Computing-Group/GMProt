from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

PHYSIO_CHEM_FILE = 'data/phyiochem.csv'

def extract_amp_features(sequences):
    """
    Extracts physicochemical features, capturing descriptors before 
    they are overwritten by subsequent modlamp calls.
    """
    # 1. Global Descriptors (MW, Charge, pI, etc.)
    glob = GlobalDescriptor(sequences)
    glob.calculate_all()
    df = pd.DataFrame(glob.descriptor, columns=glob.featurenames)

    # 2. Peptidic Descriptors (Eisenberg Scale)
    pep = PeptideDescriptor(sequences, 'eisenberg')
    
    # Calculate and capture Moment
    pep.calculate_moment() 
    df['Hydrophobic_Moment'] = pep.descriptor.flatten() 
    
    # Calculate and capture Global Hydrophobicity
    pep.calculate_global() 
    df['Global_Hydrophobicity'] = pep.descriptor.flatten() 
    
    # 3. Clean and Scale
    df = df.fillna(0)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    
    return scaled_array, df.columns.tolist()

def compute_save_physio_features():
    '''
        Computes Physiochemical Features from ecoli_normalized  Sequence CSV file and Saves it.
    '''
    df = pd.read_csv('/data/prem001/PGAT-ABPp/code/data/ecoli_normalized.csv') #
    sequences = df['sequence'].to_list()
    features_scaled, feature_names = extract_amp_features(sequences)

    # Success Check: Convert back to DataFrame for verification
    physio_df = pd.DataFrame(features_scaled, columns=feature_names)
    physio_df.insert(0, 'sequence', sequences)
    print(f"Extracted {len(feature_names)} features successfully.")
    print(physio_df.head())

    save_path = 'data/phyiochem.csv'
    physio_df.to_csv(save_path)

    print(f"PhysioChem Features saved successfully to: {save_path}.")

def load_physio_features_as_numpy():
    '''
    Returns Physico dictionary of Sequence as Key, and Numpy array as its feature values.
    '''
    df = pd.read_csv(PHYSIO_CHEM_FILE)
    #drop 'Unnamed: 0' to get only the numeric features and set index to sequence column
    #index orientation| keys = sequences value: dict:{col1: value1, col2: val2, col3: val3}
    physio_map = df.set_index('sequence').drop(columns=['Unnamed: 0'], errors='ignore').to_dict('index')

    '''
    Eg:
    Key: YPELQQDLIARLL 
    Value: {'Length': -0.6082551447299205, 'MW': -0.582454124890362, 
    'Charge': -1.8165588249838385, 'ChargeDensity': -1.7817525128767344, 'pI': -2.9343701092948087, 
    'InstabilityInd': 1.0819424035517982, 'Aromaticity': -0.3370577589352167,
    'AliphaticInd': 1.05090662530887, 'BomanInd': -0.0170077023703997,
    'HydrophRatio': 0.262069142615557, 'Hydrophobic_Moment': -0.1620249751930577,
        'Global_Hydrophobicity': 0.2418275906882444
     }

    '''
    features_dict = {}
    #Add just filtered columns | top 12 features
    for seq, feature_row_dict in physio_map.items():
        feature_row = []
        net_charge = feature_row_dict['Charge']
        m_w = feature_row_dict['MW'] #Molecular weight 
        s_len = feature_row_dict['Length'] #Seq Len | new 
        pI = feature_row_dict['pI'] #Isoeletric point 
        g_h = feature_row_dict['Global_Hydrophobicity']
        b_i = feature_row_dict['BomanInd']
        c_d = feature_row_dict['ChargeDensity']
        arm = feature_row_dict['Aromaticity'] #new
        ins_i = feature_row_dict['InstabilityInd'] #new
        h_m = feature_row_dict['Hydrophobic_Moment'] 
        al_i = feature_row_dict['AliphaticInd'] #new
        hyd_r = feature_row_dict['HydrophRatio'] #new 
        

        feature_row.extend([
           net_charge,m_w, s_len, pI, g_h, b_i, c_d, arm, ins_i, h_m, al_i, hyd_r  
        ])
        feature_row = np.array(feature_row, dtype=np.float32)
        has_invalid = ~np.isfinite(feature_row).all()
        if has_invalid:
            raise ValueError("Input Contains Nan or Infinite values.")
        

        features_dict[seq] = feature_row
    
    print(f"Length of Physico Features: {len(features_dict)}")
    return features_dict

def integrate_physio_with_mic(
    physio_csv="/data/prem001/PGAT-ABPp/code/data/phyiochem.csv",
    mic_csv="/data/prem001/PGAT-ABPp/code/data/ecoli_normalized.csv",
    output_csv="/data/prem001/PGAT-ABPp/code/data/physio_with_mic.csv"
):
    """
    Merge physico-chemical features with normalized MIC values based on sequence.

    Returns
    -------
    pd.DataFrame
        Integrated dataframe with all physico-chemical features + normalized_value
    """

    # Load data
    physio_df = pd.read_csv(physio_csv)
    mic_df = pd.read_csv(mic_csv)

    # Sanity check
    required_cols_physio = {"sequence"}
    required_cols_mic = {"sequence", "normalized_value"}

    if not required_cols_physio.issubset(physio_df.columns):
        raise ValueError("physiochem.csv must contain 'sequence' column")

    if not required_cols_mic.issubset(mic_df.columns):
        raise ValueError("ecoli_normalized.csv must contain 'sequence' and 'normalized_value' columns")

    # Merge on sequence
    merged_df = physio_df.merge(
        mic_df[["sequence", "normalized_value"]],
        on="sequence",
        how="left"  # keep all physio rows
    )

    # Report missing MIC values
    missing = merged_df["normalized_value"].isna().sum()
    if missing > 0:
        print(f" Warning: {missing} sequences have no matching MIC values")

    # Save integrated file
    if output_csv is not None:
        merged_df.to_csv(output_csv, index=False)
        print(f" Integrated file saved to: {output_csv}")

    return merged_df
def compute_physio_mic_correlation(
    csv_path="/data/prem001/PGAT-ABPp/code/data/physio_with_mic.csv",
    target_col="normalized_value"
):
    """
    Compute absolute Pearson correlation between physico-chemical features and MIC.

    Returns
    -------
    dict
        Sorted dictionary {feature_name: |correlation|}
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Drop non-feature columns
    exclude_cols = {"sequence", target_col, "Unnamed: 0"}
    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Drop rows with missing values
    df = df[feature_cols + [target_col]].dropna()

    correlations = {}

    for col in feature_cols:
        corr = df[col].corr(df[target_col], method="pearson")
        correlations[col] = abs(corr)

    # Sort by absolute correlation (descending)
    correlations_sorted = dict(
        sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    )

    return correlations_sorted

def save_dict_to_txt(corr_dict, out_path="/data/prem001/PGAT-ABPp/code/data/physio_mic_correlation.txt"):
    with open(out_path, "w") as f:
        for feature, score in corr_dict.items():
            f.write(f"{feature}\t{score:.6f}\n")
        



if __name__ == "__main__":
    compute_save_physio_features()
    # load_physio_features_as_numpy()
    # integrate_physio_with_mic()
    # corr_dict = compute_physio_mic_correlation()
    # save_dict_to_txt(corr_dict)
    

    
