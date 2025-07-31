import numpy as np
import pandas as pd
import random

from scipy.special import rel_entr
from scipy.stats import kendalltau
from sklearn.metrics import f1_score
from src.evaluator import Evaluator

class Recommender:
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 colname_user: str = 'idx_user',
                 colname_item: str = 'idx_item',
                 colname_outcome: str = 'outcome',
                 colname_prediction: str = 'pred',
                 colname_treatment: str = 'treated',
                 colname_propensity: str = 'propensity'):
        """
        Базовый класс для рекомендательных систем.
        
        :param num_users: количество пользователей
        :param num_items: количество товаров
        :param colname_user: имя колонки с ID пользователя
        :param colname_item: имя колонки с ID товара
        :param colname_outcome: имя колонки с бинарным исходом (покупка или нет)
        :param colname_prediction: имя колонки для предсказаний
        :param colname_treatment: имя колонки, указывающей, было ли промо (treatment)
        :param colname_propensity: имя колонки с пропенсити (вероятностью назначения treatment)
        """
        self.num_users = num_users
        self.num_items = num_items

        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity

    def fit(self, df: pd.DataFrame, iter: int = 100):
        """
        Метод обучения. Должен быть переопределён в подклассах.
        
        :param df: обучающая таблица
        :param iter: число итераций (если требуется)
        """
        raise NotImplementedError("Метод train должен быть реализован в подклассе.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Метод предсказания. Должен быть переопределён в подклассах.
        
        :param df: таблица с пользователями и товарами
        :return: предсказанные значения (например, вероятность клика)
        """
        raise NotImplementedError("Метод predict должен быть реализован в подклассе.")

    def recommend(self, df: pd.DataFrame, num_rec: int = 10) -> pd.DataFrame:
        raise NotImplementedError("Метод predict должен быть реализован в подклассе.")
        # """
        # Возвращает топ-N рекомендаций на пользователя.

        # :param df: входная таблица с пользователями и товарами
        # :param num_rec: число рекомендаций на одного пользователя
        # :return: таблица с колонками пользователя, товара и предсказания
        # """
        # df_pred = df.copy()
        # df_pred[self.colname_prediction] = self.predict(df_pred)

        # # Сортировка и выбор топ-N товаров для каждого пользователя
        # recommendations = (
        #     df_pred
        #     .sort_values([self.colname_user, self.colname_prediction], ascending=[True, False])
        #     .groupby(self.colname_user)
        #     .head(num_rec)
        #     .reset_index(drop=True)
        # )
        # return recommendations

    def evaluate_propensity_metrics(self, df, epsilon=0.5):
        p_pred = self.predict(df)
        p_true = df['propensity'].values
        z_true = df['treated'].values

        z_pred = (p_pred >= epsilon).astype(int)
        p_pred = np.clip(p_pred, 1e-4, 1 - 1e-4)
        p_true = np.clip(p_true, 1e-4, 1 - 1e-4)

        kld = np.mean(rel_entr(p_true, p_pred))
        tau, _ = kendalltau(p_true, p_pred)
        f1 = f1_score(z_true, z_pred)

        print(f"KLD: {kld:.4f}, Tau: {tau:.4f}, F1: {f1:.4f}")
        return {'kld': kld, 'tau': tau, 'f1': f1}