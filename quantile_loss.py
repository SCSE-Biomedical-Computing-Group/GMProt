import tensorflow as tf

class MultiQuantileLoss(tf.keras.losses.Loss):
    """
    Quantile loss for three quantiles: Q25, Q50, Q75
    Ensures Q25 < Q50 < Q75 during training.
    """

    def __init__(self, quantiles=(0.25, 0.5, 0.75), name="multi_quantile_loss", **kwargs):
        super().__init__(name=name, **kwargs)  # Accepts 'reduction' etc.
        self.quantiles = quantiles

    def call(self, y_true, y_pred):
        """
        y_true: [batch, 1]
        y_pred: [batch, 3] -> [Q25, Q50, Q75]
        """
        q25 = y_pred[:, 0:1]
        q50 = y_pred[:, 1:2]
        q75 = y_pred[:, 2:3]

        # Ensure ordering by softplus
        delta_lower = tf.nn.softplus(q50 - q25)
        delta_upper = tf.nn.softplus(q75 - q50)

        q25_corrected = q50 - delta_lower
        q75_corrected = q50 + delta_upper

        # Stack corrected predictions
        y_pred_corrected = tf.concat([q25_corrected, q50, q75_corrected], axis=-1)

        # Compute quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = y_true - y_pred_corrected[:, i:i+1]
            loss = tf.maximum(q * errors, (q - 1) * errors)
            losses.append(loss)

        return tf.reduce_mean(tf.add_n(losses))
