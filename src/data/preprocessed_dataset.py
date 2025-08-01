
from pathlib import Path
import pandas as pd
import numpy as np

def get_dataset(dataset: str, path_to_data: str):
    """
    Загружает train/vali/test CSV файлы, кодирует user/item в [0..N-1],
    формирует item_popularity — встречаемость item в позитивных событиях.
    
    :return: train_df, vali_df, test_df, num_users, num_items, num_times, item_popularity
    """
    path_to_data = Path(path_to_data)
    base = Path('data/preprocessed')
    if dataset == "CO":
        data_path = path_to_data /base / "dunn_cat_mailer_10_10_1_1/original_rp0.40"
    elif dataset == "CP":
        data_path = path_to_data /base / "dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf2.00_nr210"
    elif dataset == "PO":
        data_path = path_to_data / base / "dunn_mailer_10_10_1_1/original_rp0.90"
    elif dataset == "PP":
        data_path = path_to_data / base / "dunn_mailer_10_10_1_1/rank_rp0.90_sf2.00_nr991"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_df = pd.read_csv(data_path / "data_train.csv")
    vali_df  = pd.read_csv(data_path / "data_vali.csv")
    test_df  = pd.read_csv(data_path / "data_test.csv")

    # Перенумерация пользователей и предметов
    user_ids = pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique()
    item_ids = pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique()
    user_map = {old: new for new, old in enumerate(user_ids)}
    item_map = {old: new for new, old in enumerate(item_ids)}

    for df in (train_df, vali_df, test_df):
        df["idx_user"] = df["idx_user"].map(user_map)
        df["idx_item"] = df["idx_item"].map(item_map)

    num_users = len(user_ids)
    num_items = len(item_ids)

    # Обработка временной колонки
    if "idx_time" not in train_df.columns:
        for df in (train_df, vali_df, test_df):
            df["idx_time"] = 0
        num_times = 1
    else:
        num_times = int(train_df["idx_time"].max()) + 1

    # Обеспечиваем наличие колонок
    for df in (train_df, vali_df, test_df):
        if "propensity" not in df.columns:
            df["propensity"] = 0.0
        if "treated" not in df.columns:
            df["treated"] = 0
        if "outcome" not in df.columns:
            df["outcome"] = 0

    # Вычисляем популярность по позитивам в train
    df_pos = train_df[train_df["outcome"] > 0]
    counts = df_pos["idx_item"].value_counts().sort_index()
    item_popularity = np.zeros(num_items, dtype=float)
    item_popularity[counts.index.values] = counts.values

    return train_df, vali_df, test_df, num_users, num_items, item_popularity