import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Flatten, Dense, BatchNormalization
import tensorflow_probability as tfp
import numpy as np
from uplift.models.propcare import PropCare
from uplift.models.dlce import DLCE

import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

class PropDLCE:
    def __init__(self, num_users, num_items, args, item_popularity, freeze_propcare=True):
        self.freeze_propcare = freeze_propcare

        # инициализируем PropCare и DLCE
        self.propcare = PropCare(num_users, num_items, args, item_popularity)
        self.dlce = DLCE(num_users, num_items, args)

    def train(self, train_df, vali_df, args):
        # === Шаг 1: Обучение PropCare ===
        print("Training PropCare...")
        self.propcare.train_propensity(train_df, vali_df, args)

        # === Шаг 2: Оценка propensity на train_df и vali_df ===
        print("Predicting propensity with PropCare...")
        train_df = train_df.copy()
        vali_df = vali_df.copy()
        train_df['propensity'] = self.propcare.fit(train_df)
        vali_df['propensity'] = self.propcare.fit(vali_df)

        # === Шаг 3: Обучение DLCE ===
        if self.freeze_propcare:
            print("Training DLCE (frozen PropCare)...")
            self.dlce.fit(train_df, epochs=args.dlce_epochs)
        else:
            print("Training PropCare + DLCE jointly...")
            self._joint_train(train_df, args)

    def _joint_train(self, train_df, args):
        # делаем shared-тренировку с вычислением градиентов сразу по двум моделям
        num_items = self.propcare.item_popularity.shape[0]

        for epoch in range(args.epoch):
            items_j = np.random.randint(0, num_items, size=len(train_df))
            items_j = np.where(items_j == train_df['idx_item'], (items_j + 1) % num_items, items_j)

            dataset = tf.data.Dataset.from_tensor_slices((
                train_df['idx_user'].values.astype(np.int32),
                train_df['idx_item'].values.astype(np.int32),
                items_j.astype(np.int32),
                train_df['outcome'].values.astype(np.float32)
            )).shuffle(buffer_size=1024).batch(args.batch_size)

            for (u, i, j, y) in dataset:
                with tf.GradientTape(persistent=True) as tape:
                    # прогон через PropCare
                    _, p_i, r_i = self.propcare((u, i), training=True)
                    _, p_j, r_j = self.propcare((u, j), training=True)

                    # логика DLCE: делаем ITE через propensity
                    click_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y[:, None], p_i * r_i))

                    pop_i = tf.gather(self.propcare.item_popularity, i)
                    pop_j = tf.gather(self.propcare.item_popularity, j)
                    sgn = tf.sign(pop_i - pop_j)
                    p_diff = sgn * (p_i - p_j)
                    r_diff = sgn * (r_j - r_i)
                    weight = tf.exp(-self.propcare.exp_weight * tf.square((p_i * r_i) - (p_j * r_j)))
                    pair_loss = -tf.reduce_mean(weight * (tf.math.log_sigmoid(p_diff) + tf.math.log_sigmoid(r_diff)))

                    loss = click_loss + self.propcare.lambda_1 * pair_loss

                # применяем градиенты и к DLCE, и к PropCare
                grads = tape.gradient(loss, self.propcare.trainable_weights)
                self.propcare.optimizer.apply_gradients(zip(grads, self.propcare.trainable_weights))
                del tape  # удаляем, чтобы избежать утечек памяти

    def predict(self, df):
        return self.dlce.predict(df)


class PropDLCE_Torch:
    def __init__(self, num_users, num_items, args, item_popularity, freeze_propcare=True, device='cuda'):
        self.freeze_propcare = freeze_propcare
        self.device = device

        # Инициализация моделей
        self.propcare = PropCare(num_users, num_items, args, item_popularity, device=self.device)
        self.dlce = DLCE(num_users, num_items, 
                               dim_factor=args.dlce_dim,
                               metric=args.dlce_metric,
                               learn_rate=args.dlce_lr,
                               reg_factor=args.dlce_reg,
                               reg_bias=args.dlce_bias,
                               capping_T=args.dlce_capping_T,
                               capping_C=args.dlce_capping_C,
                               omega=args.dlce_omega,
                               xT=args.dlce_xT,
                               xC=args.dlce_xC,
                               with_bias=args.dlce_with_bias,
                               with_outcome=args.dlce_with_outcome,
                               only_treated=args.dlce_only_treated,
                               seed=args.seed,
                               device=self.device)
        self.history = []

    def fit(self, train_df, vali_df, args):
        # === 1. Обучение PropCare ===
        print("🔧 Step 1: Training PropCare...")
        self.propcare.fit(train_df, vali_df, args)

        # === 2. Предсказание propensity ===
        print("🔮 Step 2: Predicting Propensity with PropCare...")
        train_df = train_df.copy()
        vali_df = vali_df.copy()

        train_df['propensity'] = self.propcare.predict(train_df)
        vali_df['propensity'] = self.propcare.predict(vali_df)

        # === 3. Обучение DLCE ===
        if self.freeze_propcare:
            print("🎯 Step 3: Training DLCE on fixed PropCare predictions...")
            self.dlce.fit(train_df, vali_df, n_epochs=args.dlce_epochs, batch_size=args.batch_size)
        else:
            print("🔁 Step 3: Joint Training (PropCare + DLCE)...")
            self._joint_fit(train_df, vali_df, args)

    def _joint_fit(self, train_df, vali_df, args):
        print("🚀 Starting joint training of PropCare + DLCE")

        num_items = self.propcare.item_popularity.shape[0]
        users = train_df['idx_user'].values.astype(np.int64)
        items_i = train_df['idx_item'].values.astype(np.int64)
        y = train_df['outcome'].values.astype(np.float32)
        z = train_df['treated'].values.astype(np.int64)

        optimizer = torch.optim.Adam(
            list(self.propcare.parameters()) + list(self.dlce.parameters()),
            lr=args.joint_lr
        )

        for epoch in range(args.joint_epochs):
            self.propcare.train()
            self.dlce.train()

            # Негативный сэмплинг
            items_j = np.random.randint(0, num_items, size=len(train_df))
            items_j = np.where(items_j == items_i, (items_j + 1) % num_items, items_j)

            indices = np.arange(len(train_df))
            np.random.shuffle(indices)
            batch_size = args.batch_size
            num_batches = int(np.ceil(len(indices) / batch_size))
            losses = []

            for b in tqdm(range(num_batches), desc=f"Joint Epoch {epoch}", ncols=80):
                idx = indices[b * batch_size:(b + 1) * batch_size]
                u = torch.tensor(users[idx], dtype=torch.long, device=self.device)
                i = torch.tensor(items_i[idx], dtype=torch.long, device=self.device)
                j = torch.tensor(items_j[idx], dtype=torch.long, device=self.device)
                y_true = torch.tensor(y[idx], dtype=torch.float32, device=self.device)
                z_true = torch.tensor(z[idx], dtype=torch.long, device=self.device)

                # Прогон через PropCare
                _, p_i, r_i = self.propcare(u, i)
                _, p_j, r_j = self.propcare(u, j)
                p_i = p_i.detach()  # предотвращаем градиенты при обучении DLCE
                p_j = p_j.detach()

                # Click loss + pop loss (PropCare)
                click_pred = p_i * r_i
                click_loss = F.binary_cross_entropy(click_pred, y_true.unsqueeze(1))

                pop_i = self.propcare.item_popularity[i]
                pop_j = self.propcare.item_popularity[j]
                sgn = torch.sign(pop_i - pop_j).unsqueeze(1).to(self.device)
                p_diff = sgn * (p_i - p_j)
                r_diff = sgn * (r_j - r_i)
                weight = torch.exp(-self.propcare.exp_weight * (click_pred - p_j * r_j).pow(2))
                pair_loss = -torch.mean(weight * (F.logsigmoid(p_diff) + F.logsigmoid(r_diff)))

                loss_propcare = click_loss + self.propcare.lambda_1 * pair_loss

                # DLCE: готовим ITE входы
                with torch.no_grad():
                    p_i_flat = torch.clamp(p_i.squeeze(), 1e-4, 1 - 1e-4)
                    # binary Z can be derived from p_i if needed, but we use z_true
                    propensity = p_i_flat.detach()

                # DLCE score
                s_ui = self.dlce.forward(u, i, j)
                loss_dlce = self.dlce.compute_loss(s_ui, y_true, propensity, z_true)

                loss = loss_propcare + loss_dlce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            propcare_metrics = self.propcare.get_metrics(vali_df)
            dlce_metrics = self.dlce.get_metrics(vali_df)

            self.history.append({
                'epoch': len(self.history) + 1,
                'loss': np.mean(loss.item()),
                **propcare_metrics,
                **dlce_metrics
            })


            print(f"Joint Epoch {epoch} | Loss: {np.mean(loss.item()):.4f} | Metrics : {propcare_metrics}, {dlce_metrics}")

    @torch.no_grad()
    def predict(self, df, return_all=False, batch_size=4096):
        """
        Предсказания от DLCE + PropCare:
        - return_all=False: только DLCE-оценки (ранжирование)
        - return_all=True: dict с 'dlce_score', 'propensity', 'click_prob'
        """
        df = df.copy()
        
        # 1. Получаем DLCE оценку
        dlce_score = self.dlce.predict(df, batch_size=batch_size)

        if not return_all:
            return dlce_score

        # 2. Получаем p, r, p*r от PropCare
        users = torch.tensor(df['idx_user'].values, dtype=torch.long, device=self.device)
        items = torch.tensor(df['idx_item'].values, dtype=torch.long, device=self.device)

        p_vals, r_vals, click_vals = [], [], []

        self.propcare.eval()
        for i in range(0, len(users), batch_size):
            u_batch = users[i:i + batch_size]
            i_batch = items[i:i + batch_size]
            click_batch, p_batch, r_batch = self.propcare(u_batch, i_batch)
            p_vals.append(p_batch.squeeze().cpu().numpy())
            r_vals.append(r_batch.squeeze().cpu().numpy())
            click_vals.append(click_batch.squeeze().cpu().numpy())

        return {
            'dlce_score': dlce_score,
            'propensity': np.concatenate(p_vals),
            'relevance': np.concatenate(r_vals),
            'click_prob': np.concatenate(click_vals),
        }




    def plot_curve(self):
        if not hasattr(self, 'history') or len(self.history) == 0:
            print("No training history to plot.")
            return

        df = pd.DataFrame(self.history)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # === 1. Loss ===
        axes[0].plot(df['epoch'], df['loss'], marker='o')
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        # === 2. PropCare metrics ===
        propcare_cols = [c for c in df.columns if c in ['kld', 'tau', 'f1']]
        for col in propcare_cols:
            axes[1].plot(df['epoch'], df[col], label=col, marker='o')
        axes[1].set_title("Propensity Metrics (PropCare)")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True)

        # === 3. DLCE metrics ===
        dlce_cols = [c for c in df.columns if c not in ['epoch', 'loss'] + propcare_cols]
        for col in dlce_cols:
            axes[2].plot(df['epoch'], df[col], label=col, marker='o')
        axes[2].set_title("Validation Metrics (DLCE)")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

