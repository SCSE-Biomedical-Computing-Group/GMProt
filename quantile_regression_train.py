#!/usr/bin/env python3
'''
 Dual Branch Binary with Added Physico chemical Features
'''
import os
from pathlib import Path
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, kendalltau

import data_util, data_visualization
from quantile_loss import MultiQuantileLoss
from experimental_config import ExperimentConfig

class ValPearsonCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_ds):
        self.val_ds = val_ds
        self.pearson_history = []

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for x, y in self.val_ds:
            p = self.model(x, training=False).numpy()[:, 1]
            y_true.extend(y.numpy().flatten())
            y_pred.extend(p)
        r = pearsonr(y_true, y_pred)[0]

        if logs is not None:
            logs["val_Pearson"] = r

        print(f" — val_Pearson: {r:.3f}")

        self.pearson_history.append(r)



cfg = ExperimentConfig()
# ---------------------------
# GPU
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# ---------------------------
# Hyperparameters
# ---------------------------
EMB_DIM = 1024
PHYSIO_DIM = 32 #12 Features in Physicochem  | physio+Blosum62: 20+12 = 32


# ---------------------------
from Model_Dual import GraphAttentionNetwork, MultiHeadGraphAttention, GraphAttention, TransformerEncoderReadout, PartitionPadding
import physiochem_feature_extractor as PFE


CONTACT_MAP_FILE = "/data/prem001/PGAT-ABPp/code/data/contact_map.csv"
EMBEDDING_FILE = "/data/prem001/PGAT-ABPp/code/prott5/prot_t5_xl_uniref50/prott5_residue_level.npz"

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

#Set random seet for reproducibility
set_global_seed(cfg.seed)



# ============================================================
# DATASET
# ============================================================
def generator(X):
    """Generate individual examples (not ragged yet)"""
    for emb, cm, p_feat, blosum_feat, y in X:
        rows, cols = np.where(cm > 0)
        if rows.size:
            edges = np.stack([rows, cols], axis=1).astype(np.int64)
            weights = np.ones(len(rows), dtype=np.float32)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            weights = np.zeros((0,), dtype=np.float32)
        
        #p_feat: (12, ) blosum_feat:  (20,) | axis =0 becuase our features are 1D
        p_feat_updated = np.concatenate([p_feat, blosum_feat], axis=0).astype(np.float32) 
        yield emb, edges, weights, p_feat_updated, np.array([y], dtype=np.float32)


def make_dataset(X, shuffle=False):
    """Create dataset with proper ragged batching"""
    emb_spec = tf.TensorSpec(shape=[None, EMB_DIM], dtype=tf.float32)
    edges_spec = tf.TensorSpec(shape=[None, 2], dtype=tf.int64) #Per sample shape before batching
    weights_spec = tf.TensorSpec(shape=[None], dtype=tf.float32)
    
    
    physio_spec = tf.TensorSpec(shape=[PHYSIO_DIM], dtype=tf.float32)
    label_spec = tf.TensorSpec(shape=(1,), dtype=tf.float32)

    ds = tf.data.Dataset.from_generator(
        lambda: generator(X),
        output_signature=(emb_spec, edges_spec, weights_spec, physio_spec, label_spec)
    )
    
    if shuffle:
        ds = ds.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)
    
    # Ragged batch
    ds = ds.ragged_batch(cfg.batch_size)
    
    # Map batch preparation
    ds = ds.map(
        lambda emb, edges, weights, p_feat, lbl: prepare_batch(emb, edges, weights, p_feat, lbl),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    return ds


# ============================================================
# BATCH FLATTENING
# ============================================================
def prepare_batch(batched_emb, batched_edges, batched_weights, batched_physio, batched_labels):
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


    physio_flat = tf.cast(batched_physio, tf.float32)

    # Return as dictionary matching model input names
    inputs = {
        'atom_features': nodes_flat,
        'pair_indices': edges_flat,
        'edge_weights': weights_flat,
        'molecule_indicator': prot_ids,
        'physio_features': physio_flat
    }

    return inputs, labels

def predict_performance(test_ds, model_path):
    """
    Load a trained PGAT-ABPP regression model and evaluate it on the test dataset
    using advanced MIC-aware regression metrics.

    returns computed metrics as dictionary
    """

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "DualBranchGNN_PCF" : DualBranchGNN_PCF,
            "MultiQuantileLoss": MultiQuantileLoss,
            "MultiHeadGraphAttention": MultiHeadGraphAttention,
            "GraphAttention": GraphAttention,
            "TransformerEncoderReadout": TransformerEncoderReadout,
            "PartitionPadding": PartitionPadding,
        }
    )

    print("Model loaded successfully from:", model_path)
    model.summary()


    print("\nComputing test set metrics...")
    metrics = compute_metrics(
        model=model,
        dataset=test_ds
    )

    return metrics


# ============================================================
# MODEL
# ============================================================
class DualBranchGNN_PCF(keras.Model):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        

        # Use a small L2 penalty to prevent weights from exploding

        # -------- Sequence branch --------
        self.seq_dense1 = layers.Dense(cfg.seq_dense1, activation="gelu") #512
        # self.seq_bn1 = layers.BatchNormalization() #Added batch norm
        # self.seq_ln1 = layers.LayerNormalization() # Stable Norm | didnt worked
        self.seq_dropout1 = layers.Dropout(cfg.seq_dropout) 
        self.seq_dense2 = layers.Dense(cfg.seq_dense2, activation="gelu") #128
        
        # -------- Graph branch --------
        self.gnn = GraphAttentionNetwork(
            atom_dim=EMB_DIM,
            hidden_units=cfg.gnn_hidden,
            num_heads=cfg.gnn_heads,
            num_layers=cfg.gnn_layers,
            batch_size=cfg.batch_size,
            output_dim=1
        )
        self.gnn_proj = layers.Dense(cfg.gnn_dense, activation="gelu")

        # --- ADDED: Projection for Physio features (Bottleneck)
        self.physio_proj = layers.Dense(cfg.physio_proj, activation="gelu") #64

        # -------- Fusion --------
        # Update fuse1 input dimension: Sequence(128) + GNN(128) + Physio(64) = 320
        # self.fuse_norm = layers.LayerNormalization() # new 
        self.fuse1 = layers.Dense(cfg.fuse_dense, activation="gelu") #128
        # self.fuse_bn = layers.BatchNormalization() #use for stability
        # self.fuse_ln = layers.LayerNormalization()
        self.fuse_dropout = layers.Dropout(cfg.fuse_dropout)
        self.out = layers.Dense(3)

    def call(self, inputs, training=False):
        # normalize node features newly added
        nodes = inputs['atom_features'] 
        edges = inputs['pair_indices']
        weights = inputs['edge_weights']
        prot_ids = inputs['molecule_indicator']
        physio = inputs['physio_features'] # --- ADDED Physio Features---
        

        # ===== Sequence branch =====
        mean_pool = tf.math.unsorted_segment_mean(
            nodes, prot_ids, tf.reduce_max(prot_ids) + 1
        )
        max_pool = tf.math.unsorted_segment_max(
            nodes, prot_ids, tf.reduce_max(prot_ids) + 1
        )

        seq = tf.concat([mean_pool, max_pool], axis=-1)
        seq = self.seq_dense1(seq)
        seq = self.seq_dropout1(seq, training=training)
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

        # --- ADDED: Physio features branch ---
        physio_feat = self.physio_proj(physio)

        # ===== Fusion with normalization =====
        fused = tf.concat([seq, gnn, physio_feat], axis=-1)
        # fused = self.fuse_norm(fused) # new | 

        # fused = self.fuse_bn(fused, training=training) #Only Batch norm applied
        fused = self.fuse_dropout(fused, training=training)
        raw = self.out(fused)

        q25 = raw[:, 0:1]
        q50 = q25 + tf.nn.softplus(raw[:, 1:2])
        q75 = q50 + tf.nn.softplus(raw[:, 2:3])

        return tf.concat([q25, q50, q75], axis=-1)




# ============================================================
# REGRESSION METRICS
# ============================================================
def post_hoc_calibration(y_pred, y_true):
    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1, 1), y_true)
    return lr.predict(y_pred.reshape(-1, 1))


def compute_metrics(model, dataset):
    y_true, y_pred = [], []
    q25_all, q50_all, q75_all = [], [], []
    for inputs, labels in dataset:
        preds = model(inputs, training=False)
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(preds[:, 1].numpy())  # Q0.5 ONLY

        q25_all.extend(preds[:, 0])
        q50_all.extend(preds[:, 1])
        q75_all.extend(preds[:, 2])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    q25_all = np.array(q25_all)
    q75_all = np.array(q75_all)

    uncertainty = np.mean(q75_all - q25_all)
    print("*****Mean predictive uncertainty:", uncertainty)

    metrics = {
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "R2": r2_score(y_true, y_pred),
        "Pearson": pearsonr(y_true, y_pred)[0],
        "Kendall": kendalltau(y_true, y_pred)[0]
    }

    from data_visualization import plot_correlation
    plot_correlation(y_true, y_pred)


      # Load training stats if available
    try:
        with open("mic_stats.pkl", "rb") as f:
            stats = pickle.load(f)
    except:
        stats = None

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


# ============================================================
# TRAIN
# ============================================================
def execute(model_name, datasets_index=[0,1,2,3,4], result_save_path='result.txt'):
    
    #List:  (EMB, Contact_MAP, Physio, MIC)
    datasets = data_util.load_datasets(datasets_index=datasets_index)
  
    all_metrices = []
    for i, (train_f, val_f, test_f) in enumerate(datasets):
        model, metrics =  train(train_f, val_f, test_f, i, model_name)
        all_metrices.append(metrics)
    
    for i, metric in enumerate(all_metrices):
        print(f"***********\n Model {i} Metrics : **** \n {metric}")
    
    formatted_result = data_visualization.format_metrics_table(all_metrices)
    print(f"{formatted_result}")

    with open(result_save_path, "w") as f:
        f.write(formatted_result)

    return all_metrices

def train(train_f, val_f, test_f, mdl_index, model_name):
    train_ds = make_dataset(train_f, shuffle=True)
    val_ds = make_dataset(val_f)
    test_ds = make_dataset(test_f)

    model = DualBranchGNN_PCF(cfg)


    # Using the custom loss
    loss_fn = MultiQuantileLoss(quantiles=(0.25, 0.5, 0.75))
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.lr),
        loss=loss_fn,
    )
    
    model_path = Path(f"./model/{model_name}")
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True) 
    
    cfg.save_config(model_path / f"config.json") 

     
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True), #20
        ModelCheckpoint(
            f"{model_path}/model_{mdl_index}",
            save_format="tf",     
            save_best_only=True, 
            monitor='val_loss'
        ),
    ]
    callbacks.append(ValPearsonCallback(val_ds))

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    data_visualization.plot_train_val_with_pearson(history, save_path=model_path, model_index=mdl_index)

    # ---------------- Evaluation ----------------
    print("\nEvaluating on test set...")
    
  

    metrics = compute_metrics(model, test_ds)

    return model, metrics

def execute_test(model_name, datasets_index=[0, 1, 2, 3,4]):
    datasets = data_util.load_datasets(datasets_index=datasets_index)
    model_path = f"./model/{model_name}/model_{{index}}" #{{}} index will act as template

    metrics = []
    for i, dataset in enumerate(datasets):
        train, val, test = dataset
        test_ds = make_dataset(test)
        metric = predict_performance(test_ds, model_path.format(index=i))
        print(f"Model: {i} : Evaluation ****: {metric}")
        metrics.append(metric)
    
    print("*********ALL*****************")
    print(metrics)


        
    
    
    


# ================================================XZ*============
if __name__ == "__main__":
    model_name = "quantile_0"  
    #Save metrics

    datasets_index=[0]
    result_save_path = f"./model/{model_name}/result.txt"
    # metrics = execute(model_name=model_name, datasets_index=datasets_index, result_save_path=result_save_path) #[0, 1, 2,3,4]
    execute_test(model_name, datasets_index)
    

    