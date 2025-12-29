###The defination of model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        self.kernel_attention = self.add_weight(
            name="kernel_attention",
            shape=(self.units * 2, 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        """
        inputs: tuple (node_states, edges, edge_weights)
        node_states: [num_nodes, feature_dim]
        edges: [num_edges, 2]
        edge_weights: [num_edges] (float)
        """
        node_states, edges, edge_weights = inputs
        num_nodes = tf.shape(node_states)[0]
        node_states_transformed = tf.matmul(node_states, self.kernel)
        
        # Handle empty edge case with tf.cond
        def process_edges():
            # Gather node features for each edge
            node_states_expanded = tf.gather(node_states_transformed, edges)
            node_states_expanded = tf.reshape(node_states_expanded, (tf.shape(edges)[0], -1))

            # Compute attention scores
            attention_scores = tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.kernel_attention))
            attention_scores = tf.squeeze(attention_scores, -1)

            #  ADDed DROPOUT avoid textbook attention over-confidence
            # attention_scores = tf.nn.dropout(attention_scores, rate=0.1)#did not helped so commented

            # Incorporate edge weights
            attention_scores = attention_scores * edge_weights

            # Clip and exponentiate for stability
            attention_scores = tf.exp(tf.clip_by_value(attention_scores, -2, 2))

            # Sum attention scores per source node
            attention_scores_sum = tf.math.unsorted_segment_sum(
                attention_scores,
                edges[:, 0],
                num_segments=num_nodes,
            )
            
            # Gather sum for each edge (instead of repeat + bincount)
            attention_scores_sum_gathered = tf.gather(attention_scores_sum, edges[:, 0])
            
            # Normalize attention scores
            attention_scores_norm = attention_scores / (attention_scores_sum_gathered + 1e-8)

            # Aggregate neighbor features
            node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
            out = tf.math.unsorted_segment_sum(
                node_states_neighbors * attention_scores_norm[:, tf.newaxis],
                edges[:, 0],
                num_segments=num_nodes,
            )
            return out
        
        def no_edges():
            return node_states_transformed
        
        # Check if edges exist
        has_edges = tf.greater(tf.shape(edges)[0], 0)
        return tf.cond(has_edges, process_edges, no_edges)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

@tf.keras.utils.register_keras_serializable()
class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.merge_type = merge_type

        self.attention_layers = None 
    
    def build(self, input_shape):
        # Create attention layers here to ensure proper tracking
        self.attention_layers = [
            GraphAttention(self.units) for _ in range(self.num_heads)
        ]
        super().build(input_shape)

    def call(self, inputs):
        atom_features, edges, edge_weights = inputs  # add edge_weights

        outputs = [
            att([atom_features, edges, edge_weights]) for att in self.attention_layers
        ]

        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

        return tf.nn.relu(outputs)


    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "num_heads": self.num_heads,
            "merge_type": self.merge_type,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, molecule_indicator = inputs

        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )

        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)

        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )

        gather_indices = tf.where(
            tf.reduce_sum(atom_features_stacked, axis=(1, 2)) != 0
        )
        gather_indices = tf.squeeze(gather_indices, axis=-1)

        return tf.gather(atom_features_stacked, gather_indices, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "batch_size": self.batch_size,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self,
        num_heads=8,
        embed_dim=64,
        dense_dim=512,
        batch_size=32,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.batch_size = batch_size

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)

        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(
            proj_input + self.dense_proj(proj_input)
        )

        return self.average_pooling(proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "batch_size": self.batch_size,
        })
        return config

        
def GraphAttentionNetwork(atom_dim, hidden_units, num_heads, num_layers, batch_size=32, num_classes=1):
    
    # Input layers
    node_features = layers.Input((atom_dim,), dtype="float32", name="atom_features")
    pair_indices = layers.Input((2,), dtype="int32", name="pair_indices")
    edge_weights = layers.Input((), dtype="float32", name="edge_weights")  # NEW
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    
    # Preprocess features
    x = layers.Dense(hidden_units * num_heads, activation="relu")(node_features)
    
    # Multi-head graph attention layers
    for _ in range(num_layers):
        # x = MultiHeadGraphAttention(hidden_units, num_heads)([x, pair_indices, edge_weights]) + x #old

        residual = x
        x_att = MultiHeadGraphAttention(hidden_units, num_heads)([x, pair_indices, edge_weights])
        x = x_att + residual   #  No LayerNorm here
    
    # Transformer encoder readout
    x = TransformerEncoderReadout(embed_dim=hidden_units*num_heads, batch_size=batch_size)([x, molecule_indicator])
    
    x = layers.LayerNormalization()(x)   # New | Graph-level normalization

    # Output layer (linear for regression)
    outputs = layers.Dense(num_classes)(x)  # No activation for regression
    
    # Build model
    model = keras.models.Model(
        inputs=[node_features, pair_indices, edge_weights, molecule_indicator],
        outputs=outputs
    )

    
    return model