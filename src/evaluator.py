# import numpy as np

# class Evaluator():

#     def __init__(self,
#                  colname_user='idx_user', colname_item='idx_item', colname_time='idx_time',
#                  colname_outcome='outcome', colname_prediction='pred',
#                  colname_treatment='treated', colname_propensity='propensity',
#                  colname_effect='causal_effect', colname_estimate='causal_effect_estimate'):

#         self.rank_k = None
#         self.colname_user = colname_user
#         self.colname_item = colname_item
#         self.colname_time = colname_time
#         self.colname_outcome = colname_outcome
#         self.colname_prediction = colname_prediction
#         self.colname_treatment = colname_treatment
#         self.colname_propensity = colname_propensity
#         self.colname_effect = colname_effect
#         self.colname_estimate = colname_estimate

#     def get_ranking(self, df, num_rec=10):
#         df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
#         df_ranking = df.groupby(self.colname_user).head(num_rec)
#         return df_ranking

#     def get_sorted(self, df):
#         df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
#         return df

#     def capping(self, df, cap_prop=None):
#         if cap_prop is not None and cap_prop > 0:
#             bool_cap = np.logical_and(df.loc[:, self.colname_propensity] < cap_prop,
#                                       df.loc[:, self.colname_treatment] == 1)
#             if np.sum(bool_cap) > 0:
#                 df.loc[bool_cap, self.colname_propensity] = cap_prop

#             bool_cap = np.logical_and(df.loc[:, self.colname_propensity] > 1 - cap_prop,
#                                       df.loc[:, self.colname_treatment] == 0)
#             if np.sum(bool_cap) > 0:
#                 df.loc[bool_cap, self.colname_propensity] = 1 - cap_prop

#         return df

#     def clip(self, df, cap_prop=None):
#         if cap_prop is not None and cap_prop > 0:
#             pvalue = df[self.colname_propensity].values
#             pvalue = np.clip(pvalue, cap_prop, 1-cap_prop)
#             df[self.colname_propensity] = pvalue
#         return df


#     def evaluate(self, df_origin, measure, num_rec, mode = 'ASIS', cap_prop=0.0):
#         df = df_origin.copy(deep=True)
#         # print(df.head())
#         df = self.capping(df, cap_prop)
#         # df = self.clip(df, cap_prop)
#         # print(df.head())
#         df = self.get_sorted(df)
#         # print(df.head())
#         self.rank_k = num_rec

#         if 'IPS' in measure:
#             df.loc[:, self.colname_estimate] = df.loc[:, self.colname_outcome] * \
#                                         (df.loc[:, self.colname_treatment] / df.loc[:,self.colname_propensity] - \
#                                          (1 - df.loc[:, self.colname_treatment]) / (1 - df.loc[:, self.colname_propensity]))
#         if measure == 'precision':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.prec_at_k})))
#         elif measure == 'Prec':
#             df_ranking = self.get_ranking(df, num_rec=num_rec)
#             return np.nanmean(df_ranking.loc[:, self.colname_outcome].values)
#         elif measure == 'CPrec':
#             df_ranking = self.get_ranking(df, num_rec=num_rec)
#             return np.nanmean(df_ranking.loc[:, self.colname_effect].values)
#         elif measure == 'CPrecIPS':
#             df_ranking = self.get_ranking(df, num_rec=num_rec)
#             return np.nanmean(df_ranking.loc[:, self.colname_estimate].values)
#         elif measure == 'DCG':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.dcg_at_k})))
#         elif measure == 'CDCG':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.dcg_at_k})))
#         elif measure == 'CDCGIPS':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_estimate: self.dcg_at_k})))
#         elif measure == 'AR':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.ave_rank})))
#         elif measure == 'CAR':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.ave_rank})))
#         elif measure == 'CARIPS':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_estimate: self.ave_rank})))
#         elif measure == 'CARP':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.arp})))
#         elif measure == 'CARPIPS':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_estimate: self.arp})))
#         elif measure == 'CARN':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.arn})))
#         elif measure == 'CARNIPS':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_estimate: self.arn})))
#         elif measure == 'NDCG':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.ndcg_at_k})))
#         elif measure == 'hit':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.hit_at_k})))
#         elif measure == 'AUC':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_outcome: self.auc})))
#         elif measure == 'CAUC':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.gauc})))
#         elif measure == 'CAUCP':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.gaucp})))
#         elif measure == 'CAUCN':
#             return float(np.nanmean(df.groupby(self.colname_user).agg({self.colname_effect: self.gaucn})))
#         else:
#             print('measure:"' + measure + '" is not supported! ')


#     # functions for each metric
#     def prec_at_k(self, x):
#         k = min(self.rank_k, len(x))  # rank_k is global variable
#         return sum(x[:k]) / k

#     def dcg_at_k(self, x):
#         k = min(self.rank_k, len(x))  # rank_k is global variable
#         return np.sum(x[:k] / np.log2(np.arange(k) + 2))

#     def ndcg_at_k(self, x):
#         k = min(self.rank_k, len(x))  # rank_k is global variable
#         max_dcg_at_k = self.dcg_at_k(sorted(x, reverse=True))
#         if max_dcg_at_k == 0:
#             return np.nan
#         else:
#             return self.dcg_at_k(x) / max_dcg_at_k

#     def hit_at_k(self, x):
#         k = min(self.rank_k, len(x))  # rank_k is global variable
#         return float(any(x[:k] > 0))

#     def auc(self, x): # for binary (1/0)
#         len_x = len(x)
#         idx_posi = np.where(x > 0)[0]
#         len_posi = len(idx_posi)
#         len_nega = len_x - len_posi
#         if len_posi == 0 or len_nega == 0:
#             return np.nan
#         cnt_posi_before_posi = (len_posi * (len_posi - 1)) / 2
#         cnt_nega_before_posi = np.sum(idx_posi) - cnt_posi_before_posi
#         return 1 - cnt_nega_before_posi / (len_posi * len_nega)

#     def gauc(self, x): # AUC with ternary (1/0/-1) value
#         x_p = x > 0
#         x_n = x < 0
#         num_p = np.sum(x_p)
#         num_n = np.sum(x_n)
#         gauc = 0.0
#         if num_p > 0:
#             gauc += self.auc(x_p) * (num_p/(num_p + num_n))
#         if num_n > 0:
#             gauc += (1.0 - self.auc(x_n)) * (num_n/(num_p + num_n))
#         return gauc

#     def gaucp(self, x):
#         return self.auc(x > 0)

#     def gaucn(self, x):
#         return self.auc(x < 0)

#     def ave_rank(self, x):
#         len_x = len(x)
#         rank = np.arange(len_x) + 1
#         return np.mean(x * rank)

#     def arp(self, x):
#         return self.ave_rank(x > 0)
#     def arn(self, x):
#         return self.ave_rank(x < 0)


import numpy as np
import pandas as pd
import re

class Evaluator:
    def __init__(self,
                 colname_user='idx_user', colname_item='idx_item', colname_time='idx_time',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity',
                 colname_effect='causal_effect', colname_estimate='causal_effect_estimate'):

        self.rank_k = None
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_time = colname_time
        self.colname_outcome = colname_outcome
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity
        self.colname_effect = colname_effect
        self.colname_estimate = colname_estimate

    def capping(self, df, cap_prop=None):
        if cap_prop is not None and cap_prop > 0:
            treated = df[self.colname_treatment] == 1
            control = df[self.colname_treatment] == 0

            df.loc[treated & (df[self.colname_propensity] < cap_prop), self.colname_propensity] = cap_prop
            df.loc[control & (df[self.colname_propensity] > 1 - cap_prop), self.colname_propensity] = 1 - cap_prop
        return df

    def evaluate(self, df_origin, measures, mode='ASIS', cap_prop=None):
        if isinstance(measures, str):
            measures = [measures]

        df = df_origin.copy(deep=True)
        df = self.capping(df, cap_prop)

        # Предварительный расчёт IPS, если нужно
        if any('IPS' in m for m in measures):
            df[self.colname_estimate] = df[self.colname_outcome] * (
                df[self.colname_treatment] / df[self.colname_propensity] -
                (1 - df[self.colname_treatment]) / (1 - df[self.colname_propensity])
            )

        # Сортировка по пользователю и предсказанию
        df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)

        results = {}

        for measure in measures:
            match = re.match(r'([A-Za-z]+)(?:_(\d+))?', measure)
            if match:
                base_measure, k_str = match.groups()
                k = int(k_str) if k_str else None
            else:
                base_measure, k = measure, None

            self.rank_k = k
            df_ranking = df.groupby(self.colname_user).head(k) if k else None

            try:
                if base_measure == 'Prec':
                    results[measure] = np.nanmean(df_ranking[self.colname_outcome])
                elif base_measure == 'CPrec':
                    results[measure] = np.nanmean(df_ranking[self.colname_effect])
                elif base_measure == 'CPrecIPS':
                    results[measure] = np.nanmean(df_ranking[self.colname_estimate])
                elif base_measure == 'precision':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_outcome: self.prec_at_k})
                    ))
                elif base_measure == 'DCG':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_outcome: self.dcg_at_k})
                    ))
                elif base_measure == 'NDCG':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_outcome: self.ndcg_at_k})
                    ))
                elif base_measure == 'CDCG':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_effect: self.dcg_at_k})
                    ))
                elif base_measure == 'CDCGIPS':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_estimate: self.dcg_at_k})
                    ))
                elif base_measure == 'hit':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_outcome: self.hit_at_k})
                    ))
                elif base_measure == 'AR':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_outcome: self.ave_rank})
                    ))
                elif base_measure == 'CAR':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_effect: self.ave_rank})
                    ))
                elif base_measure == 'CARIPS':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_estimate: self.ave_rank})
                    ))
                elif base_measure == 'AUC':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_outcome: self.auc})
                    ))
                elif base_measure == 'CAUC':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_effect: self.gauc})
                    ))
                elif base_measure == 'CAUCP':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_effect: self.gaucp})
                    ))
                elif base_measure == 'CAUCN':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_effect: self.gaucn})
                    ))
                elif base_measure == 'CARP':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_effect: self.arp})
                    ))
                elif base_measure == 'CARPIPS':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_estimate: self.arp})
                    ))
                elif base_measure == 'CARN':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_effect: self.arn})
                    ))
                elif base_measure == 'CARNIPS':
                    results[measure] = float(np.nanmean(
                        df.groupby(self.colname_user).agg({self.colname_estimate: self.arn})
                    ))
                else:
                    results[measure] = None
                    print(f"Метрика '{measure}' не поддерживается.")
            except Exception as e:
                results[measure] = None
                print(f"Ошибка при вычислении '{measure}': {e}")

        return results

    # Метрики
    def prec_at_k(self, x): return np.mean(x[:self.rank_k]) if len(x) >= self.rank_k else np.mean(x)

    def dcg_at_k(self, x):
        k = self.rank_k if self.rank_k is not None else len(x)
        return np.sum(x[:k] / np.log2(np.arange(2, 2 + k)))

    def ndcg_at_k(self, x):
        dcg = self.dcg_at_k(x)
        idcg = self.dcg_at_k(sorted(x, reverse=True))
        return dcg / idcg if idcg > 0 else np.nan

    def hit_at_k(self, x): return float(any(x[:self.rank_k] > 0))

    def auc(self, x):
        idx_pos = np.where(x > 0)[0]
        len_pos = len(idx_pos)
        len_neg = len(x) - len_pos
        if len_pos == 0 or len_neg == 0:
            return np.nan
        cnt_pos_before_pos = len_pos * (len_pos - 1) / 2
        cnt_neg_before_pos = np.sum(idx_pos) - cnt_pos_before_pos
        return 1 - cnt_neg_before_pos / (len_pos * len_neg)

    def gauc(self, x):
        x_p, x_n = x > 0, x < 0
        num_p, num_n = np.sum(x_p), np.sum(x_n)
        if num_p + num_n == 0:
            return np.nan
        result = 0.0
        if num_p > 0: result += self.auc(x_p) * (num_p / (num_p + num_n))
        if num_n > 0: result += (1 - self.auc(x_n)) * (num_n / (num_p + num_n))
        return result

    def gaucp(self, x): return self.auc(x > 0)

    def gaucn(self, x): return self.auc(x < 0)

    def ave_rank(self, x):
        ranks = np.arange(1, len(x) + 1)
        return np.mean(x * ranks)

    def arp(self, x): return self.ave_rank(x > 0)

    def arn(self, x): return self.ave_rank(x < 0)
