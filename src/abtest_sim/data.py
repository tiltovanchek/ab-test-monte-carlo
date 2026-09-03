"""Загрузка датасета Cookie Cats.

Источник: https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing
Сырые данные в репозиторий не кладём. Сначала пробуем Kaggle API
(нужен токен ~/.kaggle/kaggle.json), если не вышло - публичное зеркало на GitHub.
"""

import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import path_in_root

KAGGLE_DATASET = "yufengsui/mobile-games-ab-testing"
MIRROR_URL = (
    "https://raw.githubusercontent.com/ryanschaub/"
    "Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv"
)
CSV_PATH = path_in_root("data/cookie_cats.csv")


def _from_kaggle(dest):
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", tmp]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError):
            return False
        with zipfile.ZipFile(next(Path(tmp).glob("*.zip"))) as zf:
            dest.write_bytes(zf.read("cookie_cats.csv"))
    return True


def _from_mirror(dest):
    with urllib.request.urlopen(MIRROR_URL, timeout=60) as resp:
        dest.write_bytes(resp.read())


def download(force=False):
    """Скачивает CSV в data/, если его ещё нет, и проверяет, что файл тот самый."""
    CSV_PATH.parent.mkdir(exist_ok=True)
    if force or not CSV_PATH.exists():
        if _from_kaggle(CSV_PATH):
            print("Скачано через Kaggle API")
        else:
            _from_mirror(CSV_PATH)
            print("Скачано с зеркала на GitHub")

    df = pd.read_csv(CSV_PATH)
    expected = ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]
    if list(df.columns) != expected or len(df) != 90_189:
        raise ValueError(f"Неожиданный файл {CSV_PATH}: {list(df.columns)}, {len(df)} строк")
    return CSV_PATH
