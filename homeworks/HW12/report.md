# HW12 - time series: temporal split, baselines and GRU forecast

## 1. Кратко: что сделано

- Dataset: `S12-hw-dataset.csv`, target: `target`, one-step forecasting.
- Temporal split used for `train / validation / test` without shuffling.
- Compared methods: `B1`, `B2`, `B3`, `R1`.
- Best method selected by validation MAE: `R1`.

## 2. Среда и воспроизводимость

- Python: 3.12 (venv `.venv`)
- pandas / numpy / scikit-learn: 3.0.1 / 2.4.3 / 1.8.0
- torch: 2.11.0
- Device: CPU
- Seed: 42
- Run: open `HW12.ipynb` and execute Run All.

## 3. Данные и постановка задачи

- Dataset: `S12-hw-dataset.csv`
- Target column: `target`
- Frequency: hourly
- Forecast horizon: 1 step
- Dataset size: 4320
- Date range: 2025-01-01 00:00:00 to 2025-06-29 23:00:00
- Extra external features: no
- Short comment: the series has oscillations and local trend with noise; no missing values in raw data.

## 4. Temporal split и признаки

### 4.1. Разбиение по времени

- Split method: first 70% train, next 15% validation, last 15% test.
- Split sizes: train=3024, validation=648, test=648.
- Why random split is wrong: it leaks future information into training and inflates metrics.

### 4.2. Признаки для baseline-моделей

- Lags: `lag_1`, `lag_7`, `lag_14`
- Rolling features: `rolling_mean_7`, `rolling_std_7` using `shift(1)`
- Calendar features: `day_of_week`, `hour`
- Missing after feature engineering: dropped initial NaN rows from lag/rolling.
- Scaling: fitted only on train data (B3 features, R1 target).

## 5. Модели и эксперименты (B1, B2, B3, R1)

- B1 (`naive-last`): forecast equals previous value.
- B2 (`moving-average`): moving average baseline.
- B3 (`ridge-lag-features`): Ridge on lag/rolling/calendar features.
- R1 (`gru-forecast`): GRU on fixed-size windows.

- Main selection metric: validation MAE
- Additional metrics: RMSE, MAPE
- GRU window_size: 24
- Batch size: 64
- Epochs: 30
- Optimizer: Adam, lr=0.001
- Best GRU checkpoint saved by best validation MAE.

## 6. Результаты

- Results table: `./artifacts/runs.csv`
- Best GRU weights: `./artifacts/best_gru.pt`
- Best GRU config: `./artifacts/best_gru_config.json`
- Split figure: `./artifacts/figures/series_split.png`
- Comparison figure: `./artifacts/figures/baselines_compare.png`
- Learning curves: `./artifacts/figures/gru_learning_curves.png`
- Final test forecast: `./artifacts/figures/best_forecast_test.png`

- Best baseline among B1/B2/B3: B3
- Best val_MAE: 5.3308
- Best val_RMSE: 6.8842
- Best val_MAPE: 3.6022
- Final test_MAE of selected model: 6.2547
- Final test_RMSE of selected model: 7.8478
- Final test_MAPE of selected model: 4.0441

## 7. Анализ

Temporal split avoids leakage and keeps evaluation realistic for forecasting. Lag and rolling features improved over naive baselines. GRU achieved the best validation MAE in the final run. Leakage risks were controlled with shift-based features and train-only scaling.

## 8. Итоговый вывод

The homework uses proper time-based validation and one-time final test evaluation. The full pipeline with baselines and GRU is reproducible and artifacts are saved.

## 9. Приложение (опционально)

Optional experiments were not included in this version.
