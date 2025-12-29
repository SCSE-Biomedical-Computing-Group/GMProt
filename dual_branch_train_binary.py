#!/usr/bin/env python3
import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, kendalltau

# ---------------------------
# GPU
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)


SEED = 42

# Python built-in random
random.seed(SEED)
# NumPy
np.random.seed(SEED)
# TensorFlow
tf.random.set_seed(SEED)

# ---------------------------
# Hyperparameters
# ---------------------------
EMB_DIM = 1024
BATCH_SIZE = 256
EPOCHS = 300
LR = 1e-4

# ---------------------------
# Imports from your codebase
# ---------------------------
from Model_Dual import GraphAttentionNetwork, MultiHeadGraphAttention, GraphAttention, TransformerEncoderReadout, PartitionPadding
import seqs_prott5 as prott5
from extract_structural_features import get_contact_map

CONTACT_MAP_FILE = "/data/prem001/PGAT-ABPp/code/data/contact_map.csv"
EMBEDDING_FILE = "/data/prem001/PGAT-ABPp/code/prott5/prot_t5_xl_uniref50/prott5_residue_level.npz"

# ============================================================
# DATA
# ============================================================
def load_features():
    '''
     Returns List[Tuple[EMB, Contact_MAP, MIC]] for sequence
    '''
    df = pd.read_csv(CONTACT_MAP_FILE)
    seqs, _, embs = prott5.load_embeddings(EMBEDDING_FILE)

    features = []
    for seq, emb in zip(seqs, embs):
        cm, mic = get_contact_map(seq, df)
        emb = np.asarray(emb, np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(-1, EMB_DIM)
        features.append((emb, cm, mic))
    return features


def split_data(features, seed=42):
    random.seed(seed)
    random.shuffle(features)
    return features[:3259], features[3259:4074], features[4074:4793]


# ============================================================
# DATASET
# ============================================================
def generator(X):
    """Generate individual examples (not ragged yet)"""
    for emb, cm, y in X:
        rows, cols = np.where(cm > 0)
        if rows.size:
            edges = np.stack([rows, cols], axis=1).astype(np.int64)
            weights = np.ones(len(rows), dtype=np.float32)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            weights = np.zeros((0,), dtype=np.float32)

        yield emb, edges, weights, np.array([y], dtype=np.float32)


def make_dataset(X, shuffle=False):
    """Create dataset with proper ragged batching"""
    emb_spec = tf.TensorSpec(shape=[None, EMB_DIM], dtype=tf.float32)
    edges_spec = tf.TensorSpec(shape=[None, 2], dtype=tf.int64) #Per sample shape before batching
    weights_spec = tf.TensorSpec(shape=[None], dtype=tf.float32)
    label_spec = tf.TensorSpec(shape=(1,), dtype=tf.float32)

    ds = tf.data.Dataset.from_generator(
        lambda: generator(X),
        output_signature=(emb_spec, edges_spec, weights_spec, label_spec)
    )
    
    if shuffle:
        ds = ds.shuffle(buffer_size=2048, seed=42)
    
    # Ragged batch
    ds = ds.ragged_batch(BATCH_SIZE)
    
    # Map batch preparation
    ds = ds.map(
        lambda emb, edges, weights, lbl: prepare_batch(emb, edges, weights, lbl),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    return ds


def compute_all_regression_metrics(model, dataset, stats):
    y_true_z, y_pred_z = [], []
    for inputs, labels in dataset:  # Changed this line
        preds = model(inputs, training=False)  # Pass the entire dict
        y_true_z.extend(labels.numpy().flatten())
        y_pred_z.extend(preds.numpy().flatten())

    y_true_z = np.array(y_true_z)
    y_pred_z = np.array(y_pred_z)

    metrics_original = {
        "RMSE_orig_z": np.sqrt(np.mean((y_true_z - y_pred_z) ** 2)),
        "R2_orig_z": r2_score(y_true_z, y_pred_z),
        "Pearson_orig_z": pearsonr(y_true_z, y_pred_z)[0],
        "Kendall_orig_z": kendalltau(y_true_z, y_pred_z)[0]
    }

    y_pred_z_cal = post_hoc_calibration(y_pred_z, y_true_z)
    y_true_logmic = y_true_z * stats["std"] + stats["mean"]
    y_pred_logmic = y_pred_z_cal * stats["std"] + stats["mean"]
    y_true_mic = 10 ** y_true_logmic
    y_pred_mic = 10 ** y_pred_logmic

    metrics_mic = {
        "RMSE_cal_MIC": np.sqrt(np.mean((y_true_mic - y_pred_mic) ** 2)),
        "R2_cal_MIC": r2_score(y_true_mic, y_pred_mic),
        "Pearson_cal_MIC": pearsonr(y_true_mic, y_pred_mic)[0],
        "Kendall_cal_MIC": kendalltau(y_true_mic, y_pred_mic)[0]
    }

    all_metrics = {**metrics_original, **metrics_mic}
    print("Regression metrics:", all_metrics)
    return all_metrics


# ============================================================
# BATCH FLATTENING
# ============================================================
def prepare_batch(batched_emb, batched_edges, batched_weights, batched_labels):
    """
    Converts a batch of ragged protein embeddings, edges, and weights into
    a flattened format suitable for GNN input.
    """
    labels = tf.cast(batched_labels, tf.float32)
    if len(labels.shape) == 1:
        labels = tf.expand_dims(labels, axis=-1)

    # Number of nodes and edges per protein
    num_nodes = batched_emb.row_lengths()
    num_edges = batched_edges.row_lengths()

    # Flatten embeddings, edges, weights to DENSE tensors
    nodes_flat = batched_emb.merge_dims(0, 1)
    edges_flat = batched_edges.merge_dims(0, 1)
    weights_flat = batched_weights.merge_dims(0, 1)

    # Compute offset for each protein to shift edge indices correctly
    offsets = tf.concat([[0], tf.cumsum(num_nodes)[:-1]], axis=0)
    edge_rowids = batched_edges.value_rowids()
    edge_offsets = tf.gather(offsets, edge_rowids)
    edges_flat = edges_flat + tf.cast(tf.expand_dims(edge_offsets, axis=-1), edges_flat.dtype)

    # Protein IDs for each node
    prot_ids = tf.repeat(tf.range(tf.shape(num_nodes)[0]), num_nodes)
    prot_ids = tf.cast(prot_ids, tf.int32)

    # Return as dictionary matching model input names
    inputs = {
        'atom_features': nodes_flat,
        'pair_indices': edges_flat,
        'edge_weights': weights_flat,
        'molecule_indicator': prot_ids
    }

    return inputs, labels

def predict_performance(test_ds, model_path):
    """
    Load a trained PGAT-ABPP regression model and evaluate it on the test dataset
    using advanced MIC-aware regression metrics.
    """

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "MultiHeadGraphAttention": MultiHeadGraphAttention,
            "GraphAttention": GraphAttention,
            "TransformerEncoderReadout": TransformerEncoderReadout,
            "PartitionPadding": PartitionPadding,
        }
    )

    print("Model loaded successfully from:", model_path)
    model.summary()

    with open("mic_stats.pkl", "rb") as f:
        train_stats = pickle.load(f)

    print("\nComputing test set metrics...")
    metrics = compute_all_regression_metrics(
        model=model,
        dataset=test_ds,
        stats=train_stats
    )

    return metrics


# ============================================================
# MODEL
# ============================================================
class DualBranchGNN(keras.Model):
    def __init__(self):
        super().__init__()

        # -------- Sequence branch --------
        self.seq_dense1 = layers.Dense(512, activation="gelu")
        self.seq_dense2 = layers.Dense(128, activation="gelu")

        # -------- Graph branch --------
        self.gnn = GraphAttentionNetwork(
            atom_dim=EMB_DIM,
            hidden_units=64,
            num_heads=4,
            num_layers=1,
            batch_size=BATCH_SIZE,
            num_classes=1
        )
        self.gnn_proj = layers.Dense(128, activation="gelu")

        # -------- Fusion --------
        self.fuse1 = layers.Dense(128, activation="gelu")
        self.out = layers.Dense(1)

    def call(self, inputs, training=False):
        # Handle both dict and tuple inputs
        if isinstance(inputs, dict):
            nodes = inputs['atom_features']
            edges = inputs['pair_indices']
            weights = inputs['edge_weights']
            prot_ids = inputs['molecule_indicator']
        else:
            nodes, edges, weights, prot_ids = inputs

        # ===== Sequence branch =====
        mean_pool = tf.math.unsorted_segment_mean(
            nodes, prot_ids, tf.reduce_max(prot_ids) + 1
        )
        max_pool = tf.math.unsorted_segment_max(
            nodes, prot_ids, tf.reduce_max(prot_ids) + 1
        )

        seq = tf.concat([mean_pool, max_pool], axis=-1)
        seq = self.seq_dense1(seq)
        seq = self.seq_dense2(seq)

        # ===== Graph branch =====
        # GNN expects dict or separate inputs
        gnn_input = {
            'atom_features': nodes,
            'pair_indices': edges,
            'edge_weights': weights,
            'molecule_indicator': prot_ids
        }
        gnn = self.gnn(gnn_input, training=training)
        gnn = self.gnn_proj(gnn)

        # ===== Fusion =====
        fused = tf.concat([seq, gnn], axis=-1)
        fused = self.fuse1(fused)
        return self.out(fused)


# ============================================================
# REGRESSION METRICS
# ============================================================
def post_hoc_calibration(y_pred, y_true):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1, 1), y_true)
    return lr.predict(y_pred.reshape(-1, 1))


def compute_metrics(model, dataset, stats=None):
    y_true, y_pred = [], []
    for inputs, labels in dataset:
        preds = model(inputs, training=False)
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(preds.numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "R2": r2_score(y_true, y_pred),
        "Pearson": pearsonr(y_true, y_pred)[0],
        "Kendall": kendalltau(y_true, y_pred)[0]
    }

    # If stats provided, compute calibrated MIC metrics
    if stats:
        y_pred_cal = post_hoc_calibration(y_pred, y_true)
        y_true_logmic = y_true * stats["std"] + stats["mean"]
        y_pred_logmic = y_pred_cal * stats["std"] + stats["mean"]
        y_true_mic = 10 ** y_true_logmic
        y_pred_mic = 10 ** y_pred_logmic

        metrics.update({
            "RMSE_MIC": np.sqrt(np.mean((y_true_mic - y_pred_mic) ** 2)),
            "R2_MIC": r2_score(y_true_mic, y_pred_mic),
            "Pearson_MIC": pearsonr(y_true_mic, y_pred_mic)[0],
            "Kendall_MIC": kendalltau(y_true_mic, y_pred_mic)[0]
        })

    return metrics

import tensorflow as tf

def soft_rank(x, epsilon=1e-6):
    """
    Differentiable soft ranking.
    Args:
        x: Tensor of shape [batch_size]
        epsilon: Smoothness parameter (smaller -> closer to true ranks)
    Returns:
        Tensor of soft ranks of shape [batch_size]
    """
    x = tf.expand_dims(x, -1)  # [batch_size, 1]
    diff = x - tf.transpose(x)  # pairwise differences
    soft_rank = tf.reduce_sum(tf.sigmoid(diff / epsilon), axis=-1) + 0.5
    return soft_rank

def huber_pearson_loss(y_true, y_pred, alpha=0.2, delta=1.0, use_rank=False, beta=0.05, epsilon=1e-2):
    """
    Hybrid Huber + Pearson + optional soft-rank loss.
    
    Args:
        y_true: Ground truth tensor [batch_size]
        y_pred: Predicted tensor [batch_size]
        alpha: Weight for Pearson term
        delta: Huber delta
        use_rank: If True, adds soft-rank term
        beta: Weight for soft-rank term
        epsilon: Soft-rank smoothness parameter
    Returns:
        Scalar loss
    """
    # --- Huber loss ---
    huber_fn = tf.keras.losses.Huber(delta=delta)
    huber = huber_fn(y_true, y_pred)
    
    # --- Normalized Pearson loss ---
    y_true_centered = y_true - tf.reduce_mean(y_true)
    y_pred_centered = y_pred - tf.reduce_mean(y_pred)
    
    y_true_norm = y_true_centered / (tf.math.reduce_std(y_true_centered) + 1e-8)
    y_pred_norm = y_pred_centered / (tf.math.reduce_std(y_pred_centered) + 1e-8)
    
    pearson_corr = tf.reduce_mean(y_true_norm * y_pred_norm)
    pearson_loss = 1.0 - pearson_corr
    
    # --- Optional soft-rank term ---
    rank_loss = 0.0
    if use_rank and beta > 0.0:
        y_true_rank = soft_rank(y_true, epsilon)
        y_pred_rank = soft_rank(y_pred, epsilon)
        rank_loss = tf.reduce_mean(tf.abs(y_true_rank - y_pred_rank))
    
    # --- Combine ---
    total_loss = huber + alpha * pearson_loss + beta * rank_loss
    return total_loss




# ============================================================
# TRAIN
# ============================================================
def train():
    #List:  (EMB, Contact_MAP, MIC)
    features = load_features()
    train_f, val_f, test_f = split_data(features)

    train_ds = make_dataset(train_f, shuffle=True)
    val_ds = make_dataset(val_f)
    test_ds = make_dataset(test_f)

    model = DualBranchGNN()


    # Using the custom loss
    loss_fn = lambda y_true, y_pred: huber_pearson_loss(
        y_true, y_pred,
        alpha=0.46,      # weight for Pearson
        delta=1.0,      # Huber delta
        use_rank=True,  # include soft-rank
        beta=0.05,      # weight for soft-rank
        epsilon=1e-2    # soft-rank smoothness
    )
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=loss_fn, #keras.losses.Huber(delta=1.0),
        metrics=[
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    callbacks = [
        EarlyStopping(monitor="val_rmse", patience=20, restore_best_weights=True),
        ModelCheckpoint(
            "dual_branch_binary_best",  # Remove .h5 extension
            save_format="tf",     # Save as TensorFlow SavedModel
            save_best_only=True, 
            monitor='val_rmse'
        ),
    ]

    print("Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    # Save final model
    model.save("dual_branch_final", save_format="tf")
    print("Model saved to 'dual_branch_final' and 'dual_branch_best'")

    # ---------------- Evaluation ----------------
    print("\nEvaluating on test set...")
    
    # Load training stats if available
    try:
        with open("mic_stats.pkl", "rb") as f:
            train_stats = pickle.load(f)
    except:
        train_stats = None
        print("Warning: mic_stats.pkl not found, skipping MIC-calibrated metrics")

    metrics = compute_metrics(model, test_ds, stats=train_stats)

    print("\n--- Test Metrics ---")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")

    # Save metrics
    with open("dual_branch_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    print("\nMetrics saved to 'dual_branch_metrics.pkl'")

    return model, metrics

# ============================================================
if __name__ == "__main__":
    train()
    #model = tf.keras.models.load_model("dual_branch_best")
    