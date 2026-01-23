# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-02.csv`
- Размер: (18000, 39)
- Целевая переменная: `target` (классы и их доли)
  - Класс 0: ~0.737
  - Класс 1: ~0.263
- Признаки: в основном числовые (`f01...f35` + `x_int_1`, `x_int_2`), столбец `id` исключён из признаков.

## 2. Protocol

- Разбиение: train/test = 80/20, `random_state=42`, `stratify=y`.
- Подбор: CV только на train (GridSearchCV).
  - Для DecisionTree / RandomForest: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, оптимизировали `roc_auc`.
  - Для GradientBoosting: `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` (уменьшено до 3 фолдов для ускорения), оптимизировали `roc_auc`.
  - Для Stacking: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` внутри `StackingClassifier`.
- Метрики: accuracy, F1, ROC-AUC.
  - accuracy — базовая метрика “доля верных ответов”;
  - F1 — важна при дисбалансе (учитывает precision/recall);
  - ROC-AUC — основная для бинарной классификации, не зависит от выбранного порога и оценивает качество ранжирования по вероятностям.
- Выбор лучшей модели: выполнялся по CV ROC-AUC на train (test использовался только для финальной оценки).

## 3. Models

Сравнивались следующие модели:

- DummyClassifier (`most_frequent`) — baseline.
- LogisticRegression — baseline через `Pipeline(StandardScaler + LogisticRegression)`.
- DecisionTreeClassifier:
  - показано переобучение дерева без ограничений;
  - контроль сложности и подбор гиперпараметров: `max_depth`, `min_samples_leaf`, `ccp_alpha` (GridSearchCV).
- RandomForestClassifier:
  - подбор гиперпараметров: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` (GridSearchCV).
- GradientBoostingClassifier:
  - подбор гиперпараметров: `n_estimators`, `learning_rate`, `max_depth`, `subsample` (GridSearchCV).
- (Опционально) StackingClassifier:
  - базовые модели: RandomForest + GradientBoosting + LogisticRegression (pipeline со scaler);
  - метамодель: LogisticRegression;
  - корректная CV-логика обучения базовых моделей (`cv=StratifiedKFold(...)`).

## 4. Results

Финальные метрики на test:

| Model | accuracy_test | f1_test | roc_auc_test |
|---|---:|---:|---:|
| Stacking | 0.9136 | 0.8262 | 0.9299 |
| RandomForest (tuned) | 0.8900 | 0.7509 | 0.9287 |
| GradientBoosting (tuned) | 0.8831 | 0.7490 | 0.9099 |
| DecisionTree (tuned) | 0.8383 | 0.6576 | 0.8371 |
| LogisticRegression | 0.8119 | 0.5607 | 0.7977 |
| DecisionTree (unlimited) | 0.8083 | 0.6380 | 0.7552 |
| Dummy (most_frequent) | 0.7375 | 0.0000 | 0.5000 |

Победитель по согласованному критерию (CV ROC-AUC на train): **RandomForest (tuned)** (CV AUC ≈ 0.9280).  
По финальным метрикам на test Stacking показал чуть более высокий ROC-AUC (0.9299) и заметно более высокий F1, однако test не использовался для выбора модели, только для итоговой оценки.

## 5. Analysis

- Устойчивость: ансамбли (RandomForest/GradientBoosting/Stacking) обычно дают меньший разброс метрик при смене `random_state`, чем одиночное дерево. Для проверки устойчивости можно сделать 5 прогонов с разными seed и сравнить разброс ROC-AUC/F1 хотя бы для 1–2 моделей.
- Ошибки: confusion matrix для лучшей модели (RandomForest) показывает баланс ошибок FP/FN и помогает понять, какие ошибки встречаются чаще (важно при умеренном дисбалансе классов).
- Интерпретация: permutation importance для RandomForest (top-15) показал, что наибольший вклад даёт признак `f16` (падение ROC-AUC ~0.044), далее `f19`, `f01`, `f07`, `f30` и др. (примерно 0.005–0.014). Это указывает, что основная часть сигнала сосредоточена в небольшом наборе признаков, а остальные дают более тонкие улучшения качества.

## 6. Conclusion

- Одиночное дерево решений легко переобучается (train accuracy = 1.0 при заметно более низком качестве на test).
- Контроль сложности дерева (например, `max_depth`, `min_samples_leaf`, `ccp_alpha`) заметно улучшает обобщающую способность.
- RandomForest (bagging + случайность по признакам) существенно снижает variance и даёт большой прирост ROC-AUC относительно одиночных моделей.
- Boosting также улучшает качество, но в данном датасете уступил RandomForest и Stacking по итоговым метрикам.
- Stacking корректно комбинирует сильные модели и на test дал лучшие значения ROC-AUC и F1, но в рамках честного протокола не использовался для выбора модели.
- Честный ML-протокол (фиксированный train/test + CV на train + единые метрики) позволяет корректно сравнивать модели без утечек.
