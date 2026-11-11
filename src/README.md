ML Pipeline – Olist Customer Satisfaction
==============================

## Data

Поместите `olist_public_dataset.csv` и `product_category_name_translation.csv` из материалов задания в `../datasets/`. 

## Setup Virtual Environment

Команды выполняются из `src/` с Python 3.11 или 3.12.

~~~
python -m venv ../.venv
source ../.venv/bin/activate
python -m pip install -r requirements.txt
~~~

В PowerShell активируйте окружение командой `..\.venv\Scripts\Activate.ps1`. Из уже подготовленного окружения можно запускать модули напрямую через `..\.venv\Scripts\python.exe -m`.

## Train

Перед обучением запустите MLflow из корня проекта командой `docker compose up -d`. Адрес трекинга по умолчанию — `http://localhost:5000`.

Если Docker недоступен, создайте папку `mlflow_store` в корне проекта. В отдельном терминале с активным окружением запустите команду из корня проекта.

~~~
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow_store/mlflow.db --default-artifact-root mlflow-artifacts:/ --serve-artifacts --artifacts-destination ./mlflow_store/artifacts
~~~

Запускаем `satisfaction.train` как модуль из папки `src/`. Запуски выполняются последовательно. Hydra сохраняет результаты в `src/artefacts/` в отдельных каталогах с датой и временем.

#### Linear Regression
~~~
python -m satisfaction.train hydra.job.chdir=True model=lr
~~~

Тестовый ROC-AUC — **0,721835**, RMSE — **1,199762**.

#### Gradient Boosting Regressor
~~~
python -m satisfaction.train hydra.job.chdir=True model=gbr
~~~

Тестовый ROC-AUC — **0,733958**, RMSE — **1,167894**. Обе модели превысили порог ROC-AUC **0,65**. Положительным классом для этой метрики считается отзыв с оценкой от 4.

## Inference

Пример для сохранённого запуска Gradient Boosting. После нового обучения подставьте каталог соответствующего запуска.

~~~
python -m satisfaction.predict \
    --model-path artefacts/2026-11-08_15-02-47/models/gbr.pkl \
    --transformer-path artefacts/2026-11-08_15-02-47/models/transformer.pkl \
    --input-path ../datasets/olist_public_dataset.csv \
    --output-path predictions.csv
~~~

Скрипт сохраняет `order_id` и `predicted_review_score`. Входному CSV не нужны поля отзыва. Таблица перевода категорий должна лежать рядом с входным файлом.

## Test
~~~
pytest tests
~~~

Результат — **28 passed**. При ошибке доступа к временному каталогу Windows передайте pytest доступную отдельную папку через `--basetemp`.

## Linter
~~~
pylint satisfaction
~~~

Оценка — **9,45/10**.

## Project Organization
    ├── configs            <- Configuration files (Hydra + dataclasses)
    ├── satisfaction       <- Source code for this project
    │   ├── __init__.py    <- Package entry
    │   ├── data           <- Dataset loading, splitting, cleaning
    │   ├── entities       <- Configuration dataclasses
    │   ├── features       <- Feature engineering and ColumnTransformer
    │   ├── models         <- Train / predict / evaluate helpers
    │   ├── train.py       <- Hydra training entrypoint
    │   └── predict.py     <- CLI script for batch inference
    │
    ├── assets             <- Screenshots and other submission assets
    ├── artefacts          <- Hydra artefacts
    │   └── ${now:%Y-%m-%d_%H-%M-%S}
    │       ├── train.log  <- Training log
    │       ├── models     <- Trained model and transformer (pickle)
    │       └── metrics    <- Test RMSE, ROC-AUC and CV metrics
    ├── tests              <- Unit and integration tests
    ├── LICENSE
    ├── README.md
    └── requirements.txt

`pytest.ini` и `requirements-lock-windows.txt` находятся в корне проекта.
