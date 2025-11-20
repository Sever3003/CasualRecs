import numpy as np
import pandas as pd
from typing import Dict
from tqdm import tqdm
from uplift.models.base import Recommender


class NeighborBase(Recommender):
    """
    Коллаборативная фильтрация: user-based или item-based.

    Поддержка: cosine и jaccard.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        colname_user: str = 'idx_user',
        colname_item: str = 'idx_item',
        colname_outcome: str = 'outcome',
        colname_prediction: str = 'pred',
        measure_simil: str = 'cosine',
        way_neighbor: str = 'user',
        num_neighbor: int = 100,
        similarity_exponent: float = 1.0,
        normalize_similarity: bool = False
    ):
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            colname_user=colname_user,
            colname_item=colname_item,
            colname_outcome=colname_outcome,
            colname_prediction=colname_prediction,
            colname_treatment='', colname_propensity=''
        )
        assert way_neighbor in ('user', 'item')
        assert measure_simil in ('cosine', 'jaccard')

        self.way_neighbor = way_neighbor
        self.measure_simil = measure_simil
        self.num_neighbor = num_neighbor
        self.similarity_exponent = similarity_exponent
        self.normalize_similarity = normalize_similarity

        self.dict_items2users: Dict[int, np.ndarray] = {}
        self.dict_users2items: Dict[int, np.ndarray] = {}
        self.dict_simil_users: Dict[int, Dict[int, float]] = {}
        self.dict_simil_items: Dict[int, Dict[int, float]] = {}

    def train(self, df: pd.DataFrame, iter: int = 1) -> None:
        df_pos = df[df[self.colname_outcome] > 0]
        user_col = self.colname_user
        item_col = self.colname_item

        self.dict_items2users = {
            i: df_pos[df_pos[item_col] == i][user_col].unique()
            for i in range(self.num_items)
        }
        self.dict_users2items = {
            u: df_pos[df_pos[user_col] == u][item_col].unique()
            for u in range(self.num_users)
        }

        if self.way_neighbor == 'user':
            for u in tqdm(range(self.num_users), desc="Computing user-user similarities"):
                self.dict_simil_users[u] = self._compute_neighbors(
                    u, self.dict_users2items, self.dict_users2items
                )
        else:
            for i in tqdm(range(self.num_items), desc="Computing item-item similarities"):
                self.dict_simil_items[i] = self._compute_neighbors(
                    i, self.dict_items2users, self.dict_items2users
                )

    def _compute_neighbors(
        self,
        target: int,
        dict_entities: Dict[int, np.ndarray],
        all_entities: Dict[int, np.ndarray]
    ) -> Dict[int, float]:
        target_set = set(dict_entities.get(target, []))
        if not target_set:
            return {}

        neighbors = {}
        for other, other_set in all_entities.items():
            if other == target or not other_set.size:
                continue
            sim = self._similarity(target_set, set(other_set))
            if sim > 0:
                neighbors[other] = sim

        # оставим только топ-N
        top_neighbors = dict(
            sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:self.num_neighbor]
        )

        # масштабируем
        if self.similarity_exponent != 1.0:
            top_neighbors = {k: v ** self.similarity_exponent for k, v in top_neighbors.items()}

        # нормализуем
        if self.normalize_similarity:
            total = sum(top_neighbors.values())
            if total > 0:
                top_neighbors = {k: v / total for k, v in top_neighbors.items()}

        return top_neighbors

    def _similarity(self, a: set, b: set) -> float:
        inter = a & b
        if not inter:
            return 0.0
        if self.measure_simil == 'jaccard':
            union = a | b
            return len(inter) / len(union)
        elif self.measure_simil == 'cosine':
            return len(inter) / (np.sqrt(len(a)) * np.sqrt(len(b)))
        return 0.0

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        preds = np.zeros(len(df), dtype=float)

        if self.way_neighbor == 'user':
            for idx, (u, i) in enumerate(zip(users, items)):
                neighbors = self.dict_simil_users.get(u, {})
                consumers = set(self.dict_items2users.get(i, []))
                preds[idx] = sum(neighbors.get(n, 0.0) for n in consumers)
        else:
            for idx, (u, i) in enumerate(zip(users, items)):
                neighbors = self.dict_simil_items.get(i, {})
                purchases = set(self.dict_users2items.get(u, []))
                preds[idx] = sum(neighbors.get(n, 0.0) for n in purchases)

        return preds
