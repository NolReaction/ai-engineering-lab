from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    missing_table,
    summarize_dataset,
    top_categories,
    flatten_summary_for_print
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(10, help="Top-K значений для категориальных признаков."),
    min_missing_share: float = typer.Option(0.1, help="Порог доли пропусков для списка проблемных колонок."),
    title: str = typer.Option("EDA Report", help="Заголовок отчёта (первая строка report.md)."),
) -> None:
    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    figs_dir = out_path / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    miss = missing_table(df)
    corr = correlation_matrix(df)
    top_cats = top_categories(df, max_columns=5, top_k=top_k_categories)

    flags = compute_quality_flags(summary, miss)

    # сохраняем таблицы
    summary_df.to_csv(out_path / "summary.csv", index=False)
    miss.to_csv(out_path / "missing.csv", index=True)
    corr.to_csv(out_path / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_path / "top_categories")

    # графики
    plot_histograms_per_column(df, figs_dir, max_columns=max_hist_columns)
    plot_missing_matrix(df, figs_dir / "missing_matrix.png")
    plot_correlation_heatmap(df, figs_dir / "correlation_heatmap.png")

    # проблемные по пропускам (колонки — в индексе)
    problematic_missing = miss[miss["missing_share"] >= min_missing_share].index.tolist()

    # report.md
    report_md = out_path / "report.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("## Параметры отчёта\n")
        f.write(f"- max_hist_columns: {max_hist_columns}\n")
        f.write(f"- top_k_categories: {top_k_categories}\n")
        f.write(f"- min_missing_share: {min_missing_share}\n\n")

        f.write("## Сводка\n")
        f.write(f"- rows: {summary.n_rows}, cols: {summary.n_cols}\n")
        f.write(f"- quality_score: {flags['quality_score']}\n\n")

        f.write("## Качество данных (эвристики)\n")
        f.write(f"- has_constant_columns: {flags['has_constant_columns']} | {flags.get('constant_columns', [])}\n")
        f.write(f"- has_suspicious_id_duplicates: {flags['has_suspicious_id_duplicates']} | {flags.get('id_columns_with_duplicates', [])}\n")
        f.write(f"- has_high_cardinality_categoricals: {flags['has_high_cardinality_categoricals']} | {flags.get('high_cardinality_categoricals', [])}\n\n")

        f.write("## Пропуски\n")
        if problematic_missing:
            f.write(f"Колонки с missing_share >= {min_missing_share}: {problematic_missing}\n\n")
        else:
            f.write("Проблемных колонок по пропускам не найдено.\n\n")

        f.write("## Артефакты\n")
        f.write("- summary.csv, missing.csv, correlation.csv\n")
        f.write("- top_categories/*.csv\n\n")

        f.write("## Графики\n")
        f.write("- figures/hist_*.png\n")
        f.write("- figures/missing_matrix.png\n")
        f.write("- figures/correlation_heatmap.png\n")

    typer.echo(f"Отчёт сгенерирован в: {out_path}")




if __name__ == "__main__":
    app()
