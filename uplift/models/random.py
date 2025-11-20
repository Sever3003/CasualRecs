import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from uplift.models.base import Recommender

class RandomBase(Recommender):
    """
    Рандомная модель — предсказывает случайные значения от 0 до 1.
    """

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users, num_items,
                         colname_user, colname_item,
                         colname_outcome, colname_prediction,
                         colname_treatment, colname_propensity)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, df, iter=1):
        pass

    def predict(self, df):
        # return torch.rand(len(df), device=self.device).cpu().numpy()
        return np.random.rand(len(df))
        # return tf.random.uniform(shape=(len(df),), minval=0.0, maxval=1.0, dtype=tf.float32).numpy()