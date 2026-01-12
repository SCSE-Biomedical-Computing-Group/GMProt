import pandas as pd
import numpy as np

# 20 standard amino acids in fixed order
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}


def _aa_freq(seq):
    """
    Compute global amino-acid composition.

    Parameters
    ----------
    seq : str
        peptide sequence.

    Returns
    -------
    vec : np.ndarray of shape (20,)
        Normalized frequency of each amino acid in the full sequence.
        Sum(vec) = 1 (approximately).
    
    Biological meaning:
        Captures overall residue bias (hydrophobicity, charge, etc.)
        without positional information.
    """
    vec = np.zeros(20, dtype=np.float32)

    for a in seq:
        if a in AA_TO_IDX:
            vec[AA_TO_IDX[a]] += 1

    return vec / max(len(seq), 1)


def _terminal_freq(seq, k=5):
    """
    Goal: compute normalized amino acid composition vectors for the first k residues (N-terminal)
     and last k residues (C-terminal) of the sequence.
    Compute N-terminal and C-terminal residue composition.


    Parameters
    ----------
    seq : str
        Protein or peptide sequence.
    k : int
        Number of residues considered at each terminus.

    Returns
    -------
    n_vec : np.ndarray of shape (20,)
        Normalized amino-acid frequencies in the first k residues (N-terminus).
    c_vec : np.ndarray of shape (20,)
        Normalized amino-acid frequencies in the last k residues (C-terminus).

    Biological meaning:
        The N-terminus controls membrane insertion,
        while the C-terminus often controls killing and binding activity.
        These distributions encode positional biochemical motifs.
    """
    n_vec = np.zeros(20, dtype=np.float32) #count amino acids in N terminal (first k residues)
    c_vec = np.zeros(20, dtype=np.float32) #count amino acids in C terminal (last k residues)

    #seq[:k] gives first k residues
    for a in seq[:k]:
        if a in AA_TO_IDX:
            n_vec[AA_TO_IDX[a]] += 1

    #seq[-k:] gives last k residues
    for a in seq[-k:]:
        if a in AA_TO_IDX:
            c_vec[AA_TO_IDX[a]] += 1

    return n_vec / k, c_vec / k


def extract_position_aware_features(sequences, k=5)->np.ndarray:
    """
    Extract position-aware residue composition (PARC) features.

    For each sequence, this builds a compact 60-D vector:
        [ Global AA composition (20),
          N-terminal composition (20),
          C-terminal composition (20) ]

    Parameters
    ----------
    sequences : list or array of str
        List of peptide/protein sequences.
    k : int
        Number of residues used for terminal statistics.

    Returns
    -------
    features : np.ndarray of shape (N, 60)
        Position-aware amino-acid feature matrix.

    Why this is powerful:
        This representation captures both what residues are present
        AND where they occur — something ProtT5, physio features,
        and sinusoidal encodings cannot do explicitly.
    """
    feats = []

    for seq in sequences:
        seq = seq.strip().upper()

        aa = _aa_freq(seq)
        n, c = _terminal_freq(seq, k)

        feats.append(np.concatenate([aa, n, c]))

    return np.array(feats)


def build_onehot_csv(input_csv, output_csv, seq_col="sequence"):
    """
    Read a CSV file containing sequences and generate a new CSV
    with compact position-aware residue features.

    Parameters
    ----------
    input_csv : str
        Path to input CSV containing a column with sequences.
    output_csv : str
        Path where the feature CSV will be written.
    seq_col : str
        Name of the column containing peptide sequences.

    Output
    ------
    A CSV with:
        sequence,
        AA_A ... AA_Y,
        N_A ... N_Y,
        C_A ... C_Y

    These features can be concatenated with:
        - physio features
        - Blosum features
        - graph embeddings
    """
    df = pd.read_csv(input_csv)
    sequences = df[seq_col].values

    X = extract_position_aware_features(sequences)
    print("Position-aware feature matrix shape:", X.shape)  # (N, 60)

    col_names = (
        [f"AA_{a}" for a in AA_LIST] +
        [f"N_{a}" for a in AA_LIST] +
        [f"C_{a}" for a in AA_LIST]
    )

    feat_df = pd.DataFrame(X, columns=col_names)
    out = pd.concat([df[[seq_col]], feat_df], axis=1)

    out.to_csv(output_csv, index=False)
    print(f"Saved position-aware one-hot features → {output_csv}")

def load_feature_csv_as_dict(feature_csv, seq_col="sequence"):
    """
    Load a CSV of position-aware features and return a dictionary mapping
    sequence -> numpy feature array.

    Parameters
    ----------
    feature_csv : str
        Path to the CSV file containing sequences and their features.
    seq_col : str
        Name of the column containing sequences.

    Returns
    -------
    dict
        {sequence (str): feature_array (np.array of shape (num_features,))}
    """
    # Load the CSV
    df = pd.read_csv(feature_csv)

    # Extract all columns except the sequence column
    feature_cols = [col for col in df.columns if col != seq_col]

    # Convert features to numpy arrays
    feature_dict = {}
    for _, row in df.iterrows():
        seq = row[seq_col]
        features = row[feature_cols].values.astype(np.float32)
        feature_dict[seq] = features

    print(f"Loaded {len(feature_dict)} sequences from {feature_csv} ")
    for seq, feat in list(feature_dict.items())[:1]:
        print(f"Sample sequence: {seq}")
        print(f"Feature shape: {feat.shape}")

    return feature_dict


if __name__ == "__main__":
    input_csv = "./data/ecoli_mic_normalized.csv"
    output_csv = "./data/position_aware_features.csv"
    # build_onehot_csv(input_csv, output_csv)
    load_feature_csv_as_dict(output_csv)