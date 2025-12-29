import tensorflow as tf
from tensorflow import keras

@keras.saving.register_keras_serializable(package="Custom")
class HuberPearsonLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.15, delta=1.0, use_rank=False, 
                 beta=0.05, epsilon=1e-2, reduction=tf.keras.losses.Reduction.AUTO,
                  name="HuberPearsonLoss"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.delta = delta
        self.use_rank = use_rank
        self.beta = beta
        self.epsilon = epsilon

        self.huber_fn = tf.keras.losses.Huber(delta=delta)

    def huber_pearson_loss(self, y_true, y_pred):
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
        
        huber = self.huber_fn(y_true, y_pred)
        
        # --- Normalized Pearson loss ---
        y_true_centered = y_true - tf.reduce_mean(y_true)
        y_pred_centered = y_pred - tf.reduce_mean(y_pred)
        
        y_true_norm = y_true_centered / (tf.math.reduce_std(y_true_centered) + 1e-8)
        y_pred_norm = y_pred_centered / (tf.math.reduce_std(y_pred_centered) + 1e-8)
        
        pearson_corr = tf.reduce_mean(y_true_norm * y_pred_norm)
        pearson_loss = 1.0 - pearson_corr
        
        # --- Optional soft-rank term ---
        rank_loss = 0.0
        if self.use_rank and self.beta > 0.0:
            y_true_rank = self.soft_rank(y_true, self.epsilon)
            y_pred_rank = self.soft_rank(y_pred, self.epsilon)
            rank_loss = tf.reduce_mean(tf.abs(y_true_rank - y_pred_rank))
        
        # --- Combine ---
        total_loss = huber + self.alpha * pearson_loss + self.beta * rank_loss
        return total_loss

    def soft_rank(self, x, epsilon=1e-6):
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

    def call(self, y_true, y_pred):
        return self.huber_pearson_loss(y_true, y_pred)
    
    def get_config(self):
        return {
            **super().get_config(),
            "alpha": self.alpha,
            "delta": self.delta,
            "use_rank": self.use_rank,
            "beta": self.beta,
            "epsilon": self.epsilon,
        }
    
