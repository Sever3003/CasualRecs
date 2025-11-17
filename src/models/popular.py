import numpy as np
from src.models.base import Recommender

class PopularBase(Recommender):
    """
    Модель PopularBase — предсказывает вероятность отклика на основе популярности товара.
    """

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users, num_items,
                         colname_user, colname_item,
                         colname_outcome, colname_prediction,
                         colname_treatment, colname_propensity)
        self.item_mean_outcome = {}

    def fit(self, df, iter=1):
        self.item_mean_outcome = df.groupby(self.colname_item)[self.colname_outcome].mean().to_dict()


    def predict(self, df):
        return df[self.colname_item].apply(lambda x: self.item_mean_outcome.get(x, 0.0)).values

    