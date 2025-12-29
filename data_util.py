import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import seqs_prott5 as prott5
from extract_structural_features import get_contact_map
import physiochem_feature_extractor as PFE

EMB_DIM = 1024
CONTACT_MAP_FILE = "/data/prem001/PGAT-ABPp/code/data/contact_map.csv"
EMBEDDING_FILE = "/data/prem001/PGAT-ABPp/code/prott5/prot_t5_xl_uniref50/prott5_residue_level.npz"
DATASET_PATH = "/data/prem001/PGAT-ABPp/code/data/five_fold/datasets.pkl"

def stratified_train_val_test_splits(
    features,
    train_size=3259,
    val_size=815,
    test_size=719,
    seed=42,
    n_bins=5,
    n_datasets=5
):
    """
    Generate multiple stratified train/val/test splits
    with fixed sizes and continuous MIC target.

    Returns:
        List of tuples:
        [(train_set, val_set, test_set), ...]
    """

    assert train_size + val_size + test_size == len(features), \
        "Split sizes must sum to total dataset size"

    mic_values = np.array([item[3] for item in features])
    indices = np.arange(len(features))

    # Quantile-based stratification bins
    bin_edges = np.percentile(
        mic_values,
        np.linspace(0, 100, n_bins + 1)[1:-1]
    )
    mic_binned = np.digitize(mic_values, bins=bin_edges)

    datasets = []

    # -------- Step 1: Multiple Train vs (Val + Test) splits --------
    sss_1 = StratifiedShuffleSplit(
        n_splits=n_datasets,
        train_size=train_size,
        test_size=val_size + test_size,
        random_state=seed
    )

    for split_id, (train_idx, temp_idx) in enumerate(
        sss_1.split(indices, mic_binned)
    ):
        # -------- Step 2: Validation vs Test --------
        mic_temp_binned = mic_binned[temp_idx]

        sss_2 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=val_size,
            test_size=test_size,
            random_state=seed + split_id  # ensure variation
        )

        val_sub_idx, test_sub_idx = next(
            sss_2.split(temp_idx, mic_temp_binned)
        )

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        train_set = [features[i] for i in train_idx]
        val_set   = [features[i] for i in val_idx]
        test_set  = [features[i] for i in test_idx]

        datasets.append((train_set, val_set, test_set))

    return datasets
# ============================================================
# DATA
# ============================================================
def load_features():
    '''
     Returns List[Tuple[EMB, Contact_MAP, MIC]] for sequence
    '''
    df = pd.read_csv(CONTACT_MAP_FILE)
    seqs, _, embs = prott5.load_embeddings(EMBEDDING_FILE)

   
    physio_dict = PFE.load_physio_features_as_numpy()
    features = []
    for seq, emb in zip(seqs, embs):
        cm, mic = get_contact_map(seq, df)
        emb = np.asarray(emb, np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(-1, EMB_DIM)
        
        physio_feature = physio_dict[seq]
        features.append((emb, cm, physio_feature, mic))
    return features

def save_datasets(save_path):
    """
    datasets: List of (train_set, val_set, test_set)
    filepath: str, e.g. 'datasets.npz'
    """
    features = load_features()
    datasets = stratified_train_val_test_splits(
        features,
        seed=42,
        n_bins=5,
        n_datasets=5
    )
    
    with open(save_path, "wb") as f:
        pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Datasets saved to {save_path}.")

def load_datasets(datasets_index=[0, 1, 2, 3, 4]):
    """
    datasets_index: Which Dataset to load 
    0: First dataset 1: Second dataset and so on.
    Returns: List [(train_set, val_set, test_set), (...)]
    """
    with open(DATASET_PATH, "rb") as f:
        datasets = pickle.load(f)


    max_idx = len(datasets) - 1
    for i in datasets_index:
        if i < 0 or i > max_idx:
            raise ValueError(
                f"Invalid dataset index {i}. Valid range: [0, {max_idx}]"
            )

    return [datasets[i] for i in datasets_index]


    

if __name__ == "__main__":
    # save_datasets(DATASET_PATH)
    datasets = load_datasets()


