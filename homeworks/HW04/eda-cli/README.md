````md
# S04 – eda_cli: HTTP-сервис качества датасетов (FastAPI)

Расширенная версия проекта `eda-cli` из Семинара 03.

К существующему CLI-приложению для EDA добавлен **HTTP-сервис на FastAPI** с эндпоинтами:

- `GET /health`
- `POST /quality`
- `POST /quality-from-csv`
- `POST /quality-flags-from-csv` (доп. эндпоинт HW04)

Используется в рамках Семинара 04 курса «Инженерия ИИ».

---

## Связь с S03

Проект в S04 основан на том же пакете `eda_cli`, что и в S03:

- сохраняется структура `src/eda_cli/` и CLI-команда `eda-cli`;
- добавлен модуль `api.py` с FastAPI-приложением;
- в зависимости добавлены `fastapi` и `uvicorn[standard]`.

Цель S04 — показать, как поверх уже написанного EDA-ядра поднять простой HTTP-сервис.

---

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему
- Браузер (для Swagger UI `/docs`) или любой HTTP-клиент:
  - `curl` / HTTP-клиент в IDE / Postman / Hoppscotch и т.п.

---

## Инициализация проекта

В папке проекта (каталог `homeworks/HW04/eda-cli`):

```bash
uv sync
````

Команда:

* создаст виртуальное окружение `.venv`;
* установит зависимости из `pyproject.toml` (включая FastAPI и Uvicorn);
* установит сам проект `eda-cli` в окружение.

---

## Запуск CLI (как в S03)

CLI остаётся доступным и в S04.

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

* `--sep` — разделитель (по умолчанию `,`);
* `--encoding` — кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

* `report.md` — основной отчёт в Markdown;
* `summary.csv` — таблица по колонкам;
* `missing.csv` — пропуски по колонкам;
* `correlation.csv` — корреляционная матрица (если есть числовые признаки);
* `top_categories/*.csv` — top-k категорий по строковым признакам;
* `hist_*.png` — гистограммы числовых колонок;
* `missing_matrix.png` — визуализация пропусков;
* `correlation_heatmap.png` — тепловая карта корреляций.

---

## Запуск HTTP-сервиса

HTTP-сервис реализован в модуле `eda_cli.api` на FastAPI.

### Запуск Uvicorn

```bash
uv run uvicorn eda_cli.api:app --reload --host 127.0.0.1 --port 8000
```

Пояснения:

* `eda_cli.api:app` — путь до объекта FastAPI `app` в модуле `eda_cli.api`;
* `--reload` — автоматический перезапуск сервера при изменении кода (удобно для разработки);
* `--host 127.0.0.1` — слушать только локально;
* `--port 8000` — порт сервиса (можно поменять при необходимости).

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## Эндпоинты сервиса

### 1) `GET /health`

Простейший health-check.

Пример:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемый ответ `200 OK` (JSON):

```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

---

### 2) `POST /quality` — заглушка по агрегированным признакам

Эндпоинт принимает агрегированные признаки датасета (размеры, доля пропусков и т.п.) и возвращает эвристическую оценку качества.

Пример вызова:

```bash
curl -X POST "http://127.0.0.1:8000/quality" \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 10000, "n_cols": 12, "max_missing_share": 0.15, "numeric_cols": 8, "categorical_cols": 4}'
```

Пример ответа `200 OK`:

```json
{
  "ok_for_model": true,
  "quality_score": 0.8,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 3.2,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 10000,
    "n_cols": 12
  }
}
```

---

### 3) `POST /quality-from-csv` — оценка качества по CSV-файлу

Эндпоинт принимает CSV-файл и внутри:

* читает его в `pandas.DataFrame`;
* вызывает функции из `eda_cli.core`:

  * `summarize_dataset`,
  * `missing_table`,
  * `compute_quality_flags`;
* возвращает оценку качества датасета в том же формате, что `/quality`.

Пример вызова:

```bash
curl -X POST "http://127.0.0.1:8000/quality-from-csv" \
  -F "file=@data/example.csv"
```

---

### 4) `POST /quality-flags-from-csv` — полный набор флагов качества (доп. эндпоинт HW04)

Дополнительный эндпоинт (вариант A): принимает CSV и возвращает **полный словарь**, который возвращает `compute_quality_flags` (включая `quality_score` и подробности вроде списков колонок).

Формат запроса:

* `multipart/form-data`
* поле `file` — CSV-файл

Пример вызова:

```bash
curl -X POST "http://127.0.0.1:8000/quality-flags-from-csv" \
  -F "file=@data/example.csv"
```

Пример ответа:

```json
{
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "max_missing_share": 0.0,
    "too_many_missing": false,
    "has_constant_columns": false,
    "constant_columns": [],
    "has_suspicious_id_duplicates": true,
    "id_columns_with_duplicates": ["user_id"],
    "has_high_cardinality_categoricals": false,
    "high_cardinality_categoricals": [],
    "quality_score": 0.8
  }
}
```

---

## Структура проекта (упрощённо)

```text
homeworks/
  HW04/
    eda-cli/
      pyproject.toml
      README.md                # этот файл
      src/
        eda_cli/
          __init__.py
          core.py              # EDA-логика, эвристики качества
          viz.py               # визуализации
          cli.py               # CLI (overview/report)
          api.py               # HTTP-сервис (FastAPI)
      tests/
        test_core.py           # тесты ядра
      data/
        example.csv            # учебный CSV для экспериментов
```

---

## Тесты

Запуск тестов:

```bash
uv run pytest -q
```

---

## Быстрый чек-лист (что должен суметь сделать проверяющий)

Из папки `homeworks/HW04/eda-cli`:

```bash
uv sync
uv run eda-cli report data/example.csv --out-dir reports_example
uv run pytest -q
uv run uvicorn eda_cli.api:app --reload --host 127.0.0.1 --port 8000
```

Далее в браузере:

* открыть Swagger UI: `http://127.0.0.1:8000/docs`
* проверить `GET /health`
* проверить `POST /quality`
* проверить `POST /quality-from-csv` (загрузить `data/example.csv`)
* проверить `POST /quality-flags-from-csv` (загрузить `data/example.csv`)

```
::contentReference[oaicite:0]{index=0}
```
