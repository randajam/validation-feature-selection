ai-ml-learning/
│
├── notebooks/
│   ├── 03-validation-feature-selection/
│   │   ├── 00_questions.ipynb                ← Теория: ответы на вопросы
│   │   ├── 01_preprocessing.ipynb            ← Чтение, очистка, генерация фичей
│   │   ├── 02_splitting_methods.ipynb        ← Реализация train/test/valid split
│   │   ├── 03_cross_validation.ipynb         ← Реализация и сравнение CV схем
│   │   ├── 04_feature_selection.ipynb        ← Lasso, correlation, permutation, SHAP
│   │   ├── 05_hyperparameter_tuning.ipynb    ← GridSearch, RandomSearch, Optuna
│   │   ├── 06_summary.ipynb                  ← Финальные метрики и сравнение методов
│   │   └── plan.md                           ← План и чек-лист выполнения
│
├── src/
│   ├── validation_feature_selection/
│   │   ├── __init__.py
│   │   ├── data_split.py                     ← функции split
│   │   ├── cross_validation.py               ← реализация KFold, Stratified, TimeSeriesSplit
│   │   ├── feature_selection.py              ← Lasso, correlation, permutation, SHAP
│   │   ├── hyperopt.py                       ← grid search, random search, optuna
│   │   ├── metrics.py                        ← метрики (MAE, RMSE, R2, etc.)
│   │   └── utils.py                          ← общие функции, reproducibility и seed
│
├── datasets/
│   └── two_sigma/                            ← датасет, как и в прошлых проектах
│
├── models/
│   └── saved_models/                         ← лучшие модели после отбора
│
└── README.md                                 ← описание проекта
