# validation_pearson.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from scipy.stats import pearsonr


class ValPearsonCallback(Callback):
    def __init__(self, val_ds, history_container):
        """
        Parameters
        ----------
        val_ds : tf.data.Dataset
            Validation dataset
        history_container : list
            External list to store Pearson values per epoch
        """
        super().__init__()
        self.val_ds = val_ds
        self.history_container = history_container

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []

        for x, y in self.val_ds:
            preds = self.model(x, training=False)
            y_true.extend(y.numpy().flatten())
            y_pred.extend(preds.numpy().flatten())

        r, _ = pearsonr(y_true, y_pred)
        self.history_container.append(r)

        # Register with Keras history (important for plotting)
        if logs is not None:
            logs["val_pearson"] = r
