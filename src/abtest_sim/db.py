"""DuckDB: представления над данными и запуск запросов из папки sql/."""

import duckdb

from .config import load_config, path_in_root

SQL_DIR = path_in_root("sql")


def connect():
    """Соединение в памяти с макросами из sql/macros.sql и двумя представлениями:
    cookie_cats - реальный датасет, sims - все results/*.parquet одной таблицей."""
    con = duckdb.connect()
    con.execute((SQL_DIR / "macros.sql").read_text(encoding="utf-8"))

    csv = path_in_root("data/cookie_cats.csv")
    if csv.exists():
        con.execute(f"CREATE VIEW cookie_cats AS SELECT * FROM read_csv_auto('{csv}')")

    results = path_in_root(load_config()["results_dir"])
    if any(results.glob("*.parquet")):
        con.execute(
            f"CREATE VIEW sims AS SELECT * FROM read_parquet('{results}/*.parquet', union_by_name = true)"
        )
    return con


def query(name, con=None):
    """Выполняет sql/<name>.sql и возвращает DataFrame."""
    con = con or connect()
    return con.sql((SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")).df()
