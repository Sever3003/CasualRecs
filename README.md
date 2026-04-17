# Аплифт-моделирование в рекомендательных системах

**Цель работы:** реализация моделей PropCare и DLCE, проведение экспериментов над этими моделями, формирование качественного и структурированного кода. Подробное описание моделей реализованных в данном коде и результаты эксперементов можно посмотреть в [самой работе](https://disk.yandex.ru/i/1JyT6X6WeNPWRQ)

## Требования

- Проект разработан и тестировался на Python 3.10 в линукс системе.
- Все зависимости перечислены в файле environment.yml. Для начала создайте окружение и скачайте туда основные зависимости:
  ```bash
  conda env create -n uplift_env -f environment.yml
  ```
  Затем активируйте окружение и скачайте туда модуль написанный в этом проекте следующим образом:
   ```bash
  conda activate uplift_env
  pip install e .
  ```
- Для обучения моделей и вычислений использовалась видеокарта NVIDIA GeForce RTX 4090
- PyTorch и TensorFlow корректно работают с RTX 4090 при установке драйверов и CUDA Toolkit (версии 11.8 или позднее).


## Датасет: Dunnhumby — The Complete Journey

Используем оригинальные данные [**Dunnhumby: The Complete Journey**](https://www.dunnhumby.com/source-files/)

### Подготовка данных

1. Скачайте с официального сайта данные и поместите их в папку: `./data/raw/`.
2. Запусти `uplift/data/data.ipynb` и выполни все ячейки (в новом окружении uplift_env).
3. Результатом станут четыре подготовленных папки с CSV‑файлами, размещённые в папке `data/preprocessed/`:

| Набор данных               | Файл                                   |
|---------------------------|----------------------------------------|
| Category‑Original         | `data/preprocessed/dunn_cat_mailer_10_10_1_1/original_rp0.40` |
| Category‑Personalized     | `data/preprocessed/dunn_cat_mailer_10_10_1_1/rank_rp0.40_sf2.00_nr210` |
| Product‑Original          | `data/preprocessed/dunn_mailer_10_10_1_1/original_rp0.90` |
| Product‑Personalized      | `data/preprocessed/dunn_mailer_10_10_1_1/rank_rp0.90_sf2.00_nr991` |

📌 **Замечание**: эти пути основаны на фактической структуре после запуска `data.ipynb`.

Код для генерации датасетов основан на реализации из репозитория [wonhyung64/causal](https://github.com/wonhyung64/causal) — где исправлена ошибка оригинальной статьи DLCE при создании персонализированных датасетов.

### Код загрузки данных перед запуском моделей

Перед тем как обучать модели, необходимо предварительно загрузить обработанные датасеты, содержащую train/vali/test разбиения и метаданные. Для этого была написана отдельная функция, которая выгружает датасет по его названию уже в нужном для нас виде:

```python
train_df, vali_df, test_df, num_users, num_items, item_pop = get_dataset(
    name, 
    path_to_data="."
)
````

где `name` принимает одно из четырёх значений:

* **CO** — Category-Original
* **CP** — Category-Personalized
* **PO** — Product-Original
* **PP** — Product-Personalized

a `path_to_data` - это путь до папки 'data'.

Функция возвращает полностью подготовленные объекты:

**train_df**, **vali_df**, **test_df** — финальные выборки в формате DataFrame
**num_users**, **num_items** — размеры пользовательского и товарного словарей
**item_pop** — вектор популярности товаров

Эти структуры данных готовы к непосредственному использованию в моделях рекомендательных систем. Ниже приведён минимальный пример корректной инициализации для датасета CO:

```python
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src import *


# Загрузка датасетa CO
train_df, vali_df, test_df, num_users, num_items, item_pop = get_dataset(
    'CO',
    path_to_data="."
)

```

## Модели

В проекте реализованы две рекомендательные модели: **PropCare** и **DLCE**.  

### PropCare

Для начала необходимо определить конфигурацию модели и задать аргументы.

#### Основные параметры модели PropCare

| Параметр | Источник / значение | Описание |
|---------|---------------------|----------|
| **dimension** | `args.dimension` | Размерность эмбеддингов пользователей и объектов. |
| **embedding_layer_units** | `args.embedding_layer_units` | Список размеров слоёв общей (shared) MLP, которая обрабатывает конкатенацию user/item эмбеддингов. |
| **estimator_layer_units** | `args.estimator_layer_units` | Архитектура MLP для двух веток: **propensity** (p) и **relevance** (r). |
| **lambda_1** | `args.lambda_1` | Коэффициент для компоненты популярностного штрафа 𝓛ₚₒₚ. |
| **lambda_2** | `args.lambda_2` *(по умолчанию `1e-4`)* | Коэффициент регуляризации склонностей к показу 𝓛ᵣₑg. |
| **lr** | `args.lr` | Learning rate (используется оптимизатор SGD). |
| **ablation_mode** | `args.ablation_mode` *(по умолчанию `"default"`)* | Режим абляции: `"NEG"`, `"S1"`, `"NO_P"`, `"NO_R"`, `"NO_P_R"`. Позволяет отключать части модели. |

---

### Пример использования PropCare

### Пример использования PropCare

Ниже приведён пример полной цепочки: загрузка датасета, инициализация PropCare, обучение, сохранение/загрузка модели, получение предсказаний и расчёт пропенсити-метрик (KLD, τ, F1).

```python
# Загрузка подготовленного датасета CO
train_df, vali_df, test_df, num_users, num_items, item_pop = get_dataset(
    "CO",
    path_to_data="."
)

# Определяем конфигурацию модели
class Args:
    dimension = 128
    embedding_layer_units = [256, 128, 64]
    estimator_layer_units = [64, 32, 16, 8]
    lambda_1 = 0.1
    lr = 0.01

args = Args()

# Инициализация модели PropCare на датасете CO
propcare_model = PropCare(
    num_users=num_users,
    num_items=num_items,
    args=args,
    item_popularity=item_pop,
    device='cuda'
)

# Обучение модели
propcare_model.fit(
    train_df,
    vali_df,
    batch_size=4096,
    epochs=25
)

# Сохранение модели
propcare_model.save_model(
    dir_path="saved_models/propcare",
    model_name="propcare_model"
)

# Загрузка модели
propcare_model = PropCare.load_model(
    dir_path="saved_models/propcare",
    model_name="propcare_model",
    device='cuda'
)

# Предсказания на тестовой выборке
preds = propcare_model.predict(test_df)

# Расчёт пропенсити-метрик (KLD, τ, F1)
value = propcare_model.get_metrics(test_df, epsilon=0.2)
```

### DLCE

Для начала необходимо определить конфигурацию модели и задать аргументы.

#### Основные параметры модели DLCE

| Параметр | Источник / значение | Описание |
|----------|---------------------|----------|
| **dim_factor** | `dim_factor` *(по умолчанию `100`)* | Размерность эмбеддингов пользователей и объектов. |
| **metric** | `metric` *(по умолчанию `'upper_bound_log'`)* | Тип метрики / целевой функции, используемой в DLCE. |
| **learn_rate** | `learn_rate` *(по умолчанию `0.001`)* | Скорость обучения (learning rate). |
| **reg_factor** | `reg_factor` *(по умолчанию `0.01`)* | L2-регуляризация для латентных факторов. |
| **reg_bias** | `reg_bias` *(по умолчанию `0.01`)* | L2-регуляризация для bias. |
| **omega** | `omega` *(по умолчанию `0.05`)* | Гиперпараметр, контролирующий степень корректировки смещения. |
| **xT** | `xT` *(по умолчанию `0.01`)* | Параметр обработки treated-части выборки. |
| **xC** | `xC` *(по умолчанию `0.01`)* | Параметр обработки control-части выборки. |
| **with_bias** | `with_bias` *(по умолчанию `True`)* | Включение user/item bias и глобального смещения. |
| **with_outcome** | `with_outcome` *(по умолчанию `True`)* | Использовать ли фактический outcome при обучении. |
| **only_treated** | `only_treated` *(по умолчанию `False`)* | Использовать ли только объекты с `treated = 1` при обучении. |
| **tau_mode** | `tau_mode` *(по умолчанию `'cips'`)* | Режим корректировки propensity: `'ips'`, `'cips'`, `'naive'`. |
| **seed** | `seed` *(по умолчанию `None`)* | Seed инициализации (если `None`, используется `42`). |
| **device** | `device` *(по умолчанию `'cuda'`)* | Устройство для обучения: `'cuda'` или `'cpu'`. |
| **measures** | `measures` *(по умолчанию ['CPrec_10', 'CPrec_100', 'CDCG_100', 'CDCG'])* | Метрики, вычисляемые при оценке модели. |

---

### Пример использования DLCE

### Пример использования DLCE

Ниже приведён пример инициализации, обучения, сохранения и загрузки модели **DLCE**  с использованием набора параметров.

```python
params = {
    "dim_factor": 100,
    "metric": "upper_bound_log",
    "learn_rate": 0.001,
    "reg_factor": 0.01,
    "reg_bias": 0.01,
    "omega": 0.05,
    "xT": 0.01,
    "xC": 0.01,
    "tau_mode": "cips",
    "with_bias": True,
    "with_outcome": True,
    "only_treated": False
}


# Инициализация DLCE на датасете CO
dlce_model = DLCE(
    num_users=num_users,
    num_items=num_items,
    device='cuda',
    **params
)

# Обучение модели на датасете CO
dlce_model.fit(
    train_df,
    vali_df,
    n_epochs=40,
    batch_size=512
)

# Сохранение модели
dlce_model.save_model(
    dir_path=f"saved_models/dlce/",
    model_name="model"
)

# Загрузка сохранённой модели
dlce_model = DLCE.load_model(
    dir_path=f"saved_models/dlce/",
    model_name="model",
    device='cuda'
)

# Предсказания
test_df[f"DLCE_pred"] = dlce_model.predict(test_df)

# Считаем метрики
evaluator = Evaluator()
evaluator.evaluate(test_df.rename(columns={"DLCE_pred": "pred"}), measures=['CPrec_10', 'CPrec_100', 'CDCG', 'CDCG_100'])
```

