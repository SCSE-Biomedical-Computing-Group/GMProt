#!/usr/bin/env python3
"""
Dual-branch GNN + Transformer for raw MIC regression
Features:
- ProtT5 embeddings
- GNN on contact map
- Physico-chemical + Blosum features
- Transformer encoder for sequence branch
- Attention-based fusion of branches
- Predict raw MIC directly
"""

import os
from pathlib import Path
import random
from dataclasses import asdict


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, kendalltau

import data_util, data_visualization
from experimental_config import ExperimentConfig
from Model_Dual import GraphAttentionNetwork,  TransformerEncoderReadout, CrossAttentionFusion
from validation_pearson import ValPearsonCallback
# --------------------------- GPU Setup ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# --------------------------- Hyperparameters ---------------------------
cfg = ExperimentConfig()
EMB_DIM = 1024
PHYSIO_DIM = cfg.physio_feature_dim + cfg.blosum_feature_dim + cfg.sinusoidal_feature_dim +cfg.postion_aware_feature_dim  #32+20+32+64=148 #using only physio physio (32) + Blosum features(20) + Sinusoidal PE(32)

# --------------------------- Reproducibility ---------------------------
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seed(cfg.seed)

# --------------------------- Dataset ---------------------------
def generator(X):
    """Yield each sample for tf.data"""
    for emb, cm, p_feat, blosum_feat, sinu_feat, pos_aware_feat, y in X:
        rows, cols = np.where(cm > 0) #return row array, col array | Each (rows[i], cols[i]) is an edge from node row[i] to node cols[i]
        
        #convert two 1D array into edge list [[0, 1],[1, 3] ] ...| shape num_edges x 2 | needed for GAT Input
        edges = np.stack([rows, cols], axis=1).astype(np.int64) if rows.size else np.zeros((0, 2), dtype=np.int64)
        weights = np.ones(len(rows), dtype=np.float32) if rows.size else np.zeros((0,), dtype=np.float32)
        p_feat_updated = np.concatenate([p_feat, blosum_feat, sinu_feat, pos_aware_feat], axis=0).astype(np.float32) 
        yield emb, edges, weights, p_feat_updated, np.array([y], dtype=np.float32)

def make_dataset(X, shuffle=False):
    emb_spec = tf.TensorSpec(shape=[None, EMB_DIM], dtype=tf.float32)
    edges_spec = tf.TensorSpec(shape=[None, 2], dtype=tf.int64)
    weights_spec = tf.TensorSpec(shape=[None], dtype=tf.float32)
    physio_spec = tf.TensorSpec(shape=[PHYSIO_DIM], dtype=tf.float32)
    label_spec = tf.TensorSpec(shape=(1,), dtype=tf.float32)

    ds = tf.data.Dataset.from_generator(
        lambda: generator(X),
        output_signature=(emb_spec, edges_spec, weights_spec, physio_spec, label_spec)
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)

    ds = ds.ragged_batch(cfg.batch_size)
    ds = ds.map(lambda emb, edges, weights, p_feat, lbl: prepare_batch(emb, edges, weights, p_feat, lbl),
                num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds

def prepare_batch(batched_emb, batched_edges, batched_weights, batched_physio, batched_labels):
    '''
        You have a batch of graphs represented as ragged tensors(variable lenghts).
        Each graph has its own set of nodes and edges. Here Node features are ProtT5 embeddings.
        Edges are derived from contact maps.
    
        It will prepare batched inputs for the batched graph neural network model.
        Adjust edge indices based on node offsets in the batch.
        Prot_ids indicate which peptide each node belongs to in the batch.

        So, instead of processing each graph individually, we can process the entire batch together.


    '''
    labels = tf.cast(batched_labels, tf.float32)
    if len(labels.shape) == 1:
        labels = tf.expand_dims(labels, axis=-1)

    num_nodes = batched_emb.row_lengths() #tf.RaggedTensor |  Shape: [batch_size, None, 1024] | (None: number of seqs in this batch)

    num_edges = batched_edges.row_lengths() #[[1, 2], [0, 2, 5], [2,3]] -> [2, 3, 2] number of edges per graph in batch
    nodes_flat = batched_emb.merge_dims(0, 1) #[batch, nodes, features] -> [total_nodes, features]
    edges_flat = batched_edges.merge_dims(0, 1)
    weights_flat = batched_weights.merge_dims(0, 1)

    #Added 0 ,  cumulative sum of num_nodes excluding last element. eg: [0, 2, 4, 5] ->[0, 2, 6, 11] 
    offsets = tf.concat([[0], tf.cumsum(num_nodes)[:-1]], axis=0) #offsets to adjust edge indices for batching
    
    edge_rowids = batched_edges.value_rowids() #Get which graph, each edge belongs to in the batch
   
    #“Pick rows from a tensor based on an index list. |  gather(input, indices)”
    edge_offsets = tf.gather(offsets, edge_rowids) #get offset for each edge based on which graph it belongs to
    edges_flat = edges_flat + tf.cast(tf.expand_dims(edge_offsets, axis=-1), edges_flat.dtype) #adjust edge indices based on node offsets in the batch
   
    prot_ids = tf.repeat(tf.range(tf.shape(num_nodes)[0]), num_nodes) #indicator for which node belongs to which graph in the batch
    prot_ids = tf.cast(prot_ids, tf.int32)
    physio_flat = tf.cast(batched_physio, tf.float32)

    inputs = {
        'atom_features': nodes_flat,
        'pair_indices': edges_flat,
        'edge_weights': weights_flat,
        'molecule_indicator': prot_ids,
        'physio_features': physio_flat
    }
    return inputs, labels

# --------------------------- Model ---------------------------
class ImprovedDualBranchGNN_AttentionFusion(keras.Model):
    """Dual-branch GNN + Transformer for raw MIC regression"""
    def __init__(self, cfg: ExperimentConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg



        # Sequence branch (data from  node-features ie. ProtT5 embeddings)
        self.seq_proj = layers.Dense(cfg.seq_dense1, activation=cfg.seq_proj_A)
        self.seq_transformer = TransformerEncoderReadout(
            num_heads=cfg.transformer_heads,
            embed_dim=cfg.seq_dense1,
            dense_dim=cfg.transformer_ff_dim,
            batch_size=cfg.batch_size
        )
        self.seq_out = layers.Dense(cfg.seq_dense2, activation=cfg.seq_out_A)
        self.seq_bottleneck = layers.Dense(cfg.seq_bottleneck_dim, activation=cfg.seq_bottleneck_A)
        
        # Graph branch
        self.gnn = GraphAttentionNetwork(
            atom_dim=EMB_DIM,
            hidden_units=cfg.gnn_hidden,
            num_heads=cfg.gnn_heads,
            num_layers=cfg.gnn_layers,
            batch_size=cfg.batch_size,
            output_dim=1
        )
        self.gnn_proj = layers.Dense(cfg.gnn_dense, activation=cfg.gnn_proj_A)
        self.gnn_bottleneck = layers.Dense(cfg.gnn_bottleneck_dim, activation=cfg.gnn_bottleneck_A) 
        
        # Physio branch
        self.physio_proj = layers.Dense(cfg.physio_proj, activation=cfg.physio_proj_A) 
        self.physio_bottleneck = layers.Dense(cfg.physio_bottleneck_dim, activation=cfg.physio_bottleneck_A) 

       #Cross attention fusion layer
        self.cross_attn_fusion = CrossAttentionFusion(
            dim=cfg.seq_bottleneck_dim,   # must match seq_feat / gnn_feat / physio_feat dim
            num_heads=4,
            dropout=0.1
        )
       
       
       # Fusion
        self.attn_dense = layers.Dense(1)
        self.fuse_norm = layers.LayerNormalization()
        self.fuse_dropout = layers.Dropout(cfg.fuse_dropout)
        
        # Output: raw MIC
        self.out = layers.Dense(1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "cfg": asdict(self.cfg)
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        cfg_dict = config.pop("cfg")
        # Convert dict back to dataclass
        cfg = ExperimentConfig(**cfg_dict)
        return cls(cfg, **config)

    def call(self, inputs, training=False):
        nodes = inputs['atom_features'] # (total_nodes, EMB_DIM): ProtT5 embeddings
        edges = inputs['pair_indices'] #binary contact map edges
        weights = inputs['edge_weights'] #binary edge weights (distance-based)
        prot_ids = inputs['molecule_indicator'] #indicator index for sequence/protein id in batch
        physio = inputs['physio_features']

        # Sequence branch
        mean_pool = tf.math.unsorted_segment_mean(nodes, prot_ids, tf.reduce_max(prot_ids) + 1) #It computes one graph-level embedding per protein by averaging all node embeddings that belong to the same protein.
        seq = self.seq_proj(mean_pool)
        seq = self.seq_transformer(seq, training=training)
        seq = self.seq_out(seq)
        seq_feat = self.seq_bottleneck(seq)

        # Graph branch
        gnn_nodes = self.gnn({'atom_features': nodes, 'pair_indices': edges, 'edge_weights': weights, 'molecule_indicator': prot_ids}, training=training)
        gnn_nodes = self.gnn_proj(gnn_nodes)
        gnn_feat = tf.math.unsorted_segment_mean(gnn_nodes, prot_ids, tf.reduce_max(prot_ids) + 1)
        gnn_feat = self.gnn_bottleneck(gnn_feat)
        
        # Physio branch
        physio_feat = self.physio_proj(physio) #Physico-chemical + Blosum features 
        physio_feat = self.physio_bottleneck(physio_feat)
       
        # Attention weighted feature fusion (how important is each branch)
        fused = self.get_attention_weighted_feature_fused(training, seq_feat, gnn_feat, physio_feat)

        #Try Cross-Attention based feature fusion (Didn't work well)
        # fused = self.get_cross_attention_weighted_feature_fused(training, seq_feat, gnn_feat, physio_feat)
        
        # Output
        return self.out(fused)

    def get_attention_weighted_feature_fused(self, training, seq_feat, gnn_feat, physio_feat):
        concat_feats = tf.stack([seq_feat, gnn_feat, physio_feat], axis=1)
        attn_scores = tf.nn.softmax(self.attn_dense(concat_feats), axis=1)
        fused = tf.reduce_sum(concat_feats * attn_scores, axis=1) #Weighted sum based on attention scores
        fused = self.fuse_norm(fused)
        fused = self.fuse_dropout(fused, training=training)
        return fused
    
    def get_cross_attention_weighted_feature_fused(self, training, seq_feat, gnn_feat, physio_feat):
        fused, attn_scores = self.cross_attn_fusion(
            seq_feat,
            gnn_feat,
            physio_feat,
            training=training

        )
        fused = self.fuse_norm(fused)
        fused = self.fuse_dropout(fused, training=training)

        return fused

# --------------------------- Metrics ---------------------------
def compute_metrics(model, dataset, model_path, model_index=0, visualize=True, write_results=True):
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

    if write_results:
        write_path = f"{model_path}/result_{model_index}.txt"
        with open(write_path, "w") as f:
            f.write(f"{metrics}\n")

        print(f"Results written to {write_path}")
    
    if visualize:
        save_path = f"{model_path}/correlation_{model_index}.png"
        data_visualization.plot_correlation(y_true, y_pred, save_path=save_path)
    return metrics

def get_model_path(model_name):
    model_path = Path(f"./model/{model_name}")
    model_path.mkdir(parents=True, exist_ok=True)

     #Save configuration Parameters
    cfg_save_path = model_path / f"config.json"
    return model_path, cfg_save_path
# --------------------------- Training ---------------------------
def train_model(train_f, val_f, test_f, model_name, mdl_index):
    train_ds = make_dataset(train_f, shuffle=True)
    val_ds = make_dataset(val_f)
    test_ds = make_dataset(test_f)

    model = ImprovedDualBranchGNN_AttentionFusion(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.lr),
        loss=tf.keras.losses.MeanSquaredError(), #tf.keras.losses.Huber(delta=1.0)
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    
    

    model_path, config_path = get_model_path(model_name)
   
    cfg.save_config(config_path) #Save configuration Parameters
    
    val_pearson_history = []

    callbacks = [
        EarlyStopping(monitor="val_rmse", mode="min", patience=cfg.patience, restore_best_weights=True),
        ModelCheckpoint(str(model_path / f"model_{mdl_index}"), save_best_only=True, monitor='val_rmse', mode='min'),
        ReduceLROnPlateau(monitor="val_rmse", factor=0.5, patience=50, min_lr=1e-6),
        ValPearsonCallback(val_ds, val_pearson_history)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=2)

    print(f"*******Model Summary****** {model.summary()}")

    history.history["val_Pearson"] = val_pearson_history
    data_visualization.plot_train_val_with_pearson(
        history,
        save_path=model_path / f"loss_curve_fold{mdl_index}.png"
    )


    metrics = compute_metrics(model, test_ds, model_path, model_index=mdl_index)
    print(f"[Fold {mdl_index}] Test metrics: {metrics}")
    return model, metrics

def load_model_and_evaluate_test(
    model_name, datasets_index=[0]):
    """
    Load a saved model from disk and evaluate on test_ds
    using the existing compute_metrics method.
    """
    mertics = []
    datasets = data_util.load_datasets(datasets_index=datasets_index)
    for model_index in datasets_index:
        model_dir = Path("model") / model_name / f"model_{model_index}"

        _, _, test_f = datasets[model_index]
        test_ds = make_dataset(test_f)

        # Load model with required custom objects
        model = tf.keras.models.load_model(
            model_dir,
            custom_objects={
                "ImprovedDualBranchGNN_AttentionFusion": ImprovedDualBranchGNN_AttentionFusion,
                "GraphAttentionNetwork": GraphAttentionNetwork,
                "TransformerEncoderReadout": TransformerEncoderReadout,
            },
            compile=False
        )

        # Reuse existing metric computation
        metric = compute_metrics(
            model=model,
            dataset=test_ds,
            model_path= f"model/{model_name}",
            model_index=model_index
        )
        mertics.append(metric)

    return mertics


# --------------------------- Execute ---------------------------
def execute(model_name, datasets_index=[0]):
    datasets = data_util.load_datasets(datasets_index=datasets_index)
    all_metrics = []
    for i, (train_f, val_f, test_f) in enumerate(datasets):
        _, metrics = train_model(train_f, val_f, test_f, model_name, i)
        all_metrics.append(metrics)
    
    model_path, config_path = get_model_path(model_name)
    data_util.save_results_table(all_metrics, filename=model_path / "metrics_results.csv")

    return all_metrics


if __name__ == "__main__":
    model_name = "attn_with_pos_aware"

    datasets_index = [0, 1, 2, 3, 4]
    metrics = execute(model_name=model_name, datasets_index=datasets_index)

    # metrics = load_model_and_evaluate_test(model_name=model_name, datasets_index=datasets_index)
    model_path, config_path = get_model_path(model_name)
    # data_util.save_results_table(metrics, filename=model_path / f"{model_name}_metrics_test.csv")

    
    print("All metrics:", metrics)
