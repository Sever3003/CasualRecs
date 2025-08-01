import numpy as np
from numpy.random import RandomState
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import Recommender
from src.evaluator import Evaluator

class DLCE_Torch(Recommender, nn.Module):
    def __init__(self, num_users, num_items,
                 dim_factor=200, metric='upper_bound_log',
                 learn_rate=0.01, reg_factor=0.01, reg_bias=0.01,
                 capping_T=0.01, capping_C=0.01,
                 omega=0.05, xT=0.01, xC=0.01,
                 with_bias=True, with_outcome=False, only_treated=False, tau_mode='cips',
                 seed=None, device='cuda',
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):

        Recommender.__init__(self, num_users, num_items,
                             colname_user, colname_item,
                             colname_outcome, colname_prediction,
                             colname_treatment, colname_propensity)
        nn.Module.__init__(self)
        self.dim_factor = dim_factor
        self.metric = metric
        self.learn_rate = learn_rate
        self.reg_factor = reg_factor
        self.reg_bias = reg_bias
        self.capping_T = capping_T
        self.capping_C = capping_C
        self.omega = omega
        self.xT = xT
        self.xC = xC
        self.with_bias = with_bias
        self.with_outcome = with_outcome
        self.only_treated = only_treated
        self.device = torch.device(device)
        self.tau_mode = tau_mode

        torch.manual_seed(seed or 42)

        self.user_factors = nn.Embedding(num_users, dim_factor)
        self.item_factors = nn.Embedding(num_items, dim_factor)

        if self.with_bias:
            self.user_biases = nn.Embedding(num_users, 1)
            self.item_biases = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))

        self.history = []
        self.to(self.device)
        self.measures = ['CPrec_10', 'CPrec_100', 'CDCG_100', 'CDCG']

    def forward(self, u, i, j):
        user_vec = self.user_factors(u)
        item_i_vec = self.item_factors(i)
        item_j_vec = self.item_factors(j)

        s_uij = (user_vec * (item_i_vec - item_j_vec)).sum(dim=1)

        if self.with_bias:
            s_uij += (self.item_biases(i).squeeze() - self.item_biases(j).squeeze())

        return s_uij

    def compute_loss(self, s_uij, y, p, z):
        pos_mask = z == 1
        neg_mask = z == 0

        if self.tau_mode == 'cips':
            # Capped IPS
            pos_weight = y[pos_mask] / torch.clamp(p[pos_mask], min=self.xT)
            neg_weight = y[neg_mask] / torch.clamp(1 - p[neg_mask], min=self.xC)

        elif self.tau_mode == 'ips':
            # IPS
            pos_weight = y[pos_mask] / p[pos_mask]
            neg_weight = y[neg_mask] / (1 - p[neg_mask])

        elif self.tau_mode == 'naive':
            pos_weight = y[pos_mask]
            neg_weight = y[neg_mask]

        else:
            raise ValueError(f"Unknown tau_mode: {self.tau_mode}")

        if self.metric == 'upper_bound_log':
            pos_loss = pos_weight * F.softplus(-self.omega * s_uij[pos_mask])
            neg_loss = neg_weight * F.softplus(self.omega * s_uij[neg_mask])
            loss = pos_loss.sum() + neg_loss.sum()

        elif self.metric == 'upper_bound_sigmoid':
            sig = torch.sigmoid(self.omega * s_uij)
            sig_inv = torch.sigmoid(-self.omega * s_uij)
            loss = torch.zeros_like(s_uij)
            loss[pos_mask] = pos_weight * sig[pos_mask] * sig_inv[pos_mask]
            loss[neg_mask] = neg_weight * sig[neg_mask] * sig_inv[neg_mask]
            loss = loss.sum()

        elif self.metric == 'upper_bound_hinge':
            loss = torch.zeros_like(s_uij)
            pos_mask_hinge = (z == 1) & (s_uij < 1 / self.omega)
            neg_mask_hinge = (z == 0) & (s_uij > -1 / self.omega)
            loss[pos_mask_hinge] = pos_weight * (1 - self.omega * s_uij[pos_mask_hinge])
            loss[neg_mask_hinge] = neg_weight * (1 + self.omega * s_uij[neg_mask_hinge])
            loss = loss.sum()

        return loss

    
    def get_metrics(self, df_vali: pd.DataFrame):
        evaluator = Evaluator()
        df_vali[f"pred"] = self.predict(df_vali)
        return evaluator.evaluate(df_vali, measures=self.measures)

    def fit(self, df_train: pd.DataFrame, df_vali: pd.DataFrame, n_epochs=10, batch_size=512):
        if self.with_outcome:
            df_train = df_train[df_train['outcome'] > 0].copy()

        if self.only_treated:
            df_train = df_train[df_train['treated'] == 1].copy()

        if self.tau_mode == 'cips':
            prop = df_train['propensity']
            treat = df_train['treated']
            df_train.loc[(treat == 1) & (prop < self.capping_T), 'propensity'] = self.capping_T
            df_train.loc[(treat == 0) & (prop > 1 - self.capping_C), 'propensity'] = 1 - self.capping_C

        u = torch.LongTensor(df_train['idx_user'].values)
        i = torch.LongTensor(df_train['idx_item'].values)
        y = torch.FloatTensor(df_train['outcome'].values)
        z = torch.LongTensor(df_train['treated'].values)
        p = torch.FloatTensor(df_train['propensity'].values)


        optimizer = torch.optim.Adam(self.parameters(), lr=self.learn_rate)

        for epoch in range(n_epochs):
            total_loss = 0.0
            self.train()

            j = torch.randint(0, self.num_items, i.shape)
            j[i == j] = (j[i == j] + 1) % self.num_items

            dataset = TensorDataset(u, i, j, y, z, p)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


            for batch in dataloader:
                u_b, i_b, j_b, y_b, z_b, p_b = [x.to(self.device) for x in batch]

                optimizer.zero_grad()
                s_uij = self.forward(u_b, i_b, j_b)
                loss = self.compute_loss(s_uij, y_b, p_b, z_b)

                reg = self.reg_factor * (
                    self.user_factors(u_b).pow(2).sum() +
                    self.item_factors(i_b).pow(2).sum() +
                    self.item_factors(j_b).pow(2).sum())
                if self.with_bias:
                    reg += self.reg_bias * (
                        self.item_biases(i_b).pow(2).sum() + self.item_biases(j_b).pow(2).sum())
                loss += reg

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            metric_values = self.get_metrics(df_vali)

            self.history.append({
                'epoch': len(self.history) + 1,
                'loss': avg_loss,
                **metric_values
            })
                                    

            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_loss:.4f}, Validation Metrics: {metric_values}")
    


    def predict(self, df: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        self.eval()
        preds = []

        users = torch.LongTensor(df['idx_user'].values).to(self.device)
        items = torch.LongTensor(df['idx_item'].values).to(self.device)

        num_samples = len(df)
        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                u_batch = users[start:end]
                i_batch = items[start:end]

                pred_batch = (self.user_factors(u_batch) * self.item_factors(i_batch)).sum(dim=1)

                if self.with_bias:
                    pred_batch += (
                        self.user_biases(u_batch).squeeze() +
                        self.item_biases(i_batch).squeeze() +
                        self.global_bias
                    )

                preds.append(pred_batch.cpu())

        return torch.cat(preds).numpy()


    def plot_curve(self):
        if not hasattr(self, 'history') or len(self.history) == 0:
            print("No training history to plot.")
            return

        history_df = pd.DataFrame(self.history)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # График лосса
        axes[0].plot(history_df['epoch'], history_df['loss'], marker='o')
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        metric_cols = [col for col in history_df.columns if col not in ['epoch', 'loss']]
        for metric in metric_cols:
            axes[1].plot(history_df['epoch'], history_df[metric], label=metric, marker='o')

        axes[1].set_title("Validation Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Value")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
