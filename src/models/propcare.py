
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models.base import Recommender
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import kendalltau
from sklearn.metrics import f1_score
from scipy.stats import entropy

import json
import os

class PropCare(Recommender, nn.Module):
    def __init__(self, num_users, num_items, args, item_popularity: np.ndarray, device='cuda'):
        Recommender.__init__(self,
                             num_users=num_users,
                             num_items=num_items,
                             colname_user='idx_user',
                             colname_item='idx_item',
                             colname_outcome='outcome',
                             colname_prediction='pred',
                             colname_treatment='treated',
                             colname_propensity='propensity')
        nn.Module.__init__(self)
        
        self.seed = getattr(args, "seed", 42)
        torch.manual_seed(self.seed)


        self.device = device
        self.dimension = args.dimension
        self.estimator_layer_units = args.estimator_layer_units
        self.embedding_layer_units = args.embedding_layer_units
        self.lambda_1 = args.lambda_1
        self.lambda_2 = getattr(args, "lambda_2", 1e-4)
        self.lr = args.lr
        self.ablation_mode = getattr(args, "ablation_mode", "default")  # "NEG", "S1", "NO_P", "NO_R", "NO_P_R"

        self.norm_mean = None
        self.norm_std = None

        self.u_emb = nn.Embedding(num_users, self.dimension)
        self.i_emb = nn.Embedding(num_items, self.dimension)

        # shared MLP
        layers = []
        dim_in = self.dimension * 2
        for units in self.embedding_layer_units:
            layers.append(nn.Linear(dim_in, units))
            layers.append(nn.LeakyReLU())
            dim_in = units
        self.shared_mlp = nn.Sequential(*layers)

        # ветка p (propensity)
        self.p_layers = nn.ModuleList()
        self.p_bn = nn.ModuleList()
        for units in self.estimator_layer_units:
            self.p_layers.append(nn.Linear(dim_in, units))
            self.p_bn.append(nn.BatchNorm1d(units))
            dim_in = units
        self.p_out = nn.Linear(dim_in, 1)

        # ветка r (relevance)
        dim_in = self.embedding_layer_units[-1]
        self.r_layers = nn.ModuleList()
        self.r_bn = nn.ModuleList()
        for units in self.estimator_layer_units:
            self.r_layers.append(nn.Linear(dim_in, units))
            self.r_bn.append(nn.BatchNorm1d(units))
            dim_in = units
        self.r_out = nn.Linear(dim_in, 1)

        self.alpha = nn.Parameter(torch.tensor(0.2, dtype=torch.float32, device=self.device))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=self.device))

        self.exp_weight = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.register_buffer("item_popularity", torch.tensor(item_popularity, dtype=torch.float32))

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.to(self.device)

        self.history = []

    def forward(self, u, i):
        u = u.to(self.device)
        i = i.to(self.device)

        eu = self.u_emb(u)
        ei = self.i_emb(i)
        x = torch.cat([eu, ei], dim=1)
        x = self.shared_mlp(x)

        # ветка p
        p = x
        for layer, bn in zip(self.p_layers, self.p_bn):
            p = F.leaky_relu(layer(p))
            p = bn(p)
        p = torch.sigmoid(self.p_out(p))
        p = torch.clamp(p, 1e-4, 1 - 1e-4)

        # ветка r
        r = x
        for layer, bn in zip(self.r_layers, self.r_bn):
            r = F.leaky_relu(layer(r))
            r = bn(r)
        r = torch.sigmoid(self.r_out(r))
        r = torch.clamp(r, 1e-4, 1 - 1e-4)

        return p * r, p, r

    def _train_step(self, u, i, j, y):
        self.train()
        self.optimizer.zero_grad()

        u, i, j, y = u.to(self.device), i.to(self.device), j.to(self.device), y.to(self.device)

        _, p_i, r_i = self(u, i)
        _, p_j, r_j = self(u, j)

        # Основная бинарная клик-вероятность (p * r)
        click_pred = p_i * r_i
        click_loss = F.binary_cross_entropy(click_pred, y.unsqueeze(1))

        # Разности популярности
        pop_i = self.item_popularity[i]
        pop_j = self.item_popularity[j]
        sgn = torch.sign(pop_i - pop_j).unsqueeze(1).to(self.device)

        if self.ablation_mode == "NEG":
            sgn = -sgn

        p_diff = sgn * (p_i - p_j)
        r_diff = sgn * (r_j - r_i)

        # Веса для парной функции потерь
        if self.ablation_mode == "S1":
            weight = 1.0
        else:
            weight = torch.exp(-self.exp_weight * (click_pred - p_j * r_j).pow(2))

        # Парная функция потерь (в зависимости от режима)
        if self.ablation_mode == "NO_P":
            pair_loss = -torch.mean(weight * F.logsigmoid(r_diff))
        elif self.ablation_mode == "NO_R":
            pair_loss = -torch.mean(weight * F.logsigmoid(p_diff))
        elif self.ablation_mode == "NO_P_R":
            pair_loss = torch.tensor(0.0, device=self.device)
        else:  # default, NEG, S1
            pair_loss = -torch.mean(weight * (F.logsigmoid(p_diff) + F.logsigmoid(r_diff)))

        # KL-регуляризация через бета-распределение
        beta_dist = torch.distributions.Beta(self.alpha, self.beta)
        q_i = torch.sort(beta_dist.sample(p_i.shape).to(self.device), dim=0).values
        q_j = torch.sort(beta_dist.sample(p_j.shape).to(self.device), dim=0).values
        reg = F.kl_div(p_i.log(), q_i, reduction='batchmean') + F.kl_div(p_j.log(), q_j, reduction='batchmean')

        # Общая функция потерь
        loss = click_loss + self.lambda_1 * pair_loss + self.lambda_2 * reg
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def fit(self, train_df, vali_df, batch_size=4096, epochs=10):
        best_corr = -np.inf
        best_state = None

        num_items = self.item_popularity.shape[0]

        for epoch in range(epochs):
            self.train()
            users = train_df['idx_user'].values.astype(np.int64)
            items_i = train_df['idx_item'].values.astype(np.int64)
            y = train_df['outcome'].values.astype(np.float32)

            # Негативный сэмплинг
            items_j = np.random.randint(0, num_items, size=len(train_df))
            items_j = np.where(items_j == items_i, (items_j + 1) % num_items, items_j)

            indices = np.arange(len(train_df))
            np.random.shuffle(indices)
            num_batches = int(np.ceil(len(indices) / batch_size))

            losses = []
            for b in tqdm(range(num_batches), desc=f"Epoch {epoch}", unit="batch", ncols=80):
                idx = indices[b * batch_size:(b + 1) * batch_size]
                u = torch.tensor(users[idx], dtype=torch.long)
                i = torch.tensor(items_i[idx], dtype=torch.long)
                j = torch.tensor(items_j[idx], dtype=torch.long)
                y_true = torch.tensor(y[idx], dtype=torch.float32)

                loss = self._train_step(u, i, j, y_true)
                losses.append(loss)

            # Валидация
            self.eval()

            # обновляем mean и std для предсказания z
            with torch.no_grad():
                _, p_train, _ = self.predict(train_df, return_components=True)
                self.norm_mean = float(p_train.mean())
                self.norm_std = float(p_train.std())


            metric_values = self.get_metrics(vali_df)
            print(f"Epoch {epoch}: train loss = {np.mean(losses):.4f} | val corr = {metric_values}")

            self.history.append({
                'epoch': len(self.history) + 1,
                'loss': np.mean(losses),
                **metric_values
            })

            # if metric_values['kld'] > best_corr:
            #     best_corr = metric_values['kld']
            #     best_state = self.state_dict()
        
        # if best_state is not None:
        #     self.load_state_dict(best_state)
        #     print("✅ Loaded best model based on validation correlation.")


        return self

    @torch.no_grad()
    def predict(self, df, batch_size=5096, return_components=False):
        self.eval()
        users = torch.tensor(df['idx_user'].values, dtype=torch.long, device=self.device)
        items = torch.tensor(df['idx_item'].values, dtype=torch.long, device=self.device)

        preds, props, rels = [], [], []

        for i in range(0, len(users), batch_size):
            u_batch = users[i:i + batch_size]
            i_batch = items[i:i + batch_size]
            click, p, r = self(u_batch, i_batch)

            preds.append(click.squeeze().cpu().numpy())
            if return_components:
                props.append(p.squeeze().cpu().numpy())
                rels.append(r.squeeze().cpu().numpy())

        if return_components:
            return (
                np.concatenate(preds),
                np.concatenate(props),
                np.concatenate(rels)
            )
        else:
            return np.concatenate(preds)
        
    @torch.no_grad()
    def predict_z(self, df, epsilon=0.2, batch_size=4096):
        """
        Предсказывает z_hat на основе нормализованной p_hat, используя сохранённые mean/std по трейну.
        """
        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("Normalization stats not set. Run fit() first or provide mean/std manually.")

        _, p_pred, _ = self.predict(df, batch_size=batch_size, return_components=True)
        p_norm = (p_pred - self.norm_mean) / self.norm_std
        z_hat = (p_norm >= epsilon).astype(int)
        return z_hat



    def get_metrics(self, df, epsilon=0.2):  # epsilon задаётся как в статье
        p_pred = self.predict(df)
        p_true = df['propensity'].values
        z_true = df['treated'].values

        # Клипуем для стабильности
        # p_pred = np.clip(p_pred, 1e-4, 1 - 1e-4)
        # p_true = np.clip(p_true, 1e-4, 1 - 1e-4)

        z_pred = self.predict_z(df, epsilon=epsilon, batch_size=5096)

        # Метрики
        kld = entropy(p_true, p_pred)
        tau, _ = kendalltau(p_true, p_pred)
        f1 = f1_score(z_true, z_pred)

        return {'kld': kld, 'tau': tau, 'f1': f1}

    def plot_curve(self):
        if not hasattr(self, 'history') or len(self.history) == 0:
            print("No training history to plot.")
            return
        
        history_df = pd.DataFrame(self.history)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        axes[0].plot(history_df['epoch'], history_df['loss'], marker='o')
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        if 'kld' in history_df.columns:
            axes[1].plot(history_df['epoch'], history_df['kld'], marker='o', color='tab:orange')
            axes[1].set_title("KL Divergence")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("KLD")
            axes[1].grid(True)
        else:
            axes[1].axis('off')

        if 'tau' in history_df.columns:
            axes[2].plot(history_df['epoch'], history_df['tau'], marker='o', color='tab:green')
            axes[2].set_title("Kendall Tau")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Tau")
            axes[2].grid(True)
        else:
            axes[2].axis('off')

        if 'f1' in history_df.columns:
            axes[3].plot(history_df['epoch'], history_df['f1'], marker='o', color='tab:red')
            axes[3].set_title("F1 Score")
            axes[3].set_xlabel("Epoch")
            axes[3].set_ylabel("F1")
            axes[3].grid(True)
        else:
            axes[3].axis('off')

        plt.tight_layout()
        plt.show()


    def save_model(self, dir_path="saved_models/propcare", model_name="propcare_model"):
        path = os.path.join(dir_path, model_name)
        os.makedirs(path, exist_ok=True)

        weights_path = os.path.join(path, "weights.pt")
        torch.save(self.state_dict(), weights_path)

        config = {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "dimension": self.dimension,
            "embedding_layer_units": self.embedding_layer_units,
            "estimator_layer_units": self.estimator_layer_units,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "lr": self.lr,
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "device": str(self.device),
            "seed": int(self.seed) if self.seed is not None else None
        }

        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        popularity_path = os.path.join(path, "item_popularity.npy")
        np.save(popularity_path, self.item_popularity.cpu().numpy())

        if hasattr(self, "history") and self.history:
            history_path = os.path.join(path, "history.csv")
            pd.DataFrame(self.history).to_csv(history_path, index=False)

        print(f"Model saved to: {path}")

    @classmethod
    def load_model(cls, dir_path="saved_models/propcare", model_name="propcare_model", device="cuda"):
        path = os.path.join(dir_path, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Путь {path} не существует.")

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        config["device"] = device
        config_args = type('Args', (), config)()

        popularity_path = os.path.join(path, "item_popularity.npy")
        item_popularity = np.load(popularity_path)

        model = cls(
            num_users=config["num_users"],
            num_items=config["num_items"],
            args=config_args,
            item_popularity=item_popularity,
            device=device
        )

        model.norm_mean = config.get("norm_mean", None)
        model.norm_std = config.get("norm_std", None)

        weights_path = os.path.join(path, "weights.pt")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()

        history_path = os.path.join(path, "history.csv")
        if os.path.exists(history_path):
            model.history = pd.read_csv(history_path).to_dict(orient="records")

        return model
