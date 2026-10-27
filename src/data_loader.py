"""Load the UCI Student Performance mathematics dataset."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import requests
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

def load_student_data(path: str | Path = "data/raw/student-mat.csv") -> pd.DataFrame:
    """Read local CSV or download the public UCI archive when absent."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, sep=";")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            names = archive.namelist()
            csv_name = next((n for n in names if n.endswith("student-mat.csv")), None)
            if csv_name is None:
                raise FileNotFoundError("student-mat.csv tidak ditemukan dalam arsip UCI.")
            content = archive.read(csv_name)
        path.write_bytes(content)
        return pd.read_csv(path, sep=";")
    except Exception as exc:
        raise RuntimeError(
            "Dataset tidak dapat diunduh otomatis. Unduh arsip dari "
            "https://archive.ics.uci.edu/dataset/320/student+performance, "
            "ekstrak student-mat.csv, lalu simpan di data/raw/student-mat.csv. "
            f"Detail: {exc}"
        ) from exc
