import pandas as pd
import csv
from io import StringIO


def robust_csv_reader(uploaded_file):
    """
    Robust CSV reader for Django uploads.
    - Auto-detects delimiter (, ; \t)
    - Handles EU decimals
    - Converts numeric columns properly
    - Leaves text columns (e.g. SMILES) untouched
    """

    # 1️⃣ Read content from Django UploadedFile
    try:
        content = uploaded_file.read().decode("utf-8", errors="replace")
        uploaded_file.seek(0)
    except Exception:
        raise ValueError("Unable to read uploaded file.")

    if not content.strip():
        raise ValueError("CSV file is empty.")

    # 2️⃣ Detect delimiter
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(content[:2048], delimiters=",;\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    # 3️⃣ Read using correct decimal logic
    if delimiter == ";":
        df = pd.read_csv(StringIO(content), delimiter=";", decimal=",")
    else:
        df = pd.read_csv(StringIO(content), delimiter=",")

    # 4️⃣ Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # 5️⃣ Convert numeric columns safely
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                # Keep text columns unchanged (e.g. SMILES)
                pass

    return df