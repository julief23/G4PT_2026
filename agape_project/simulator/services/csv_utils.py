import pandas as pd
import csv

def robust_csv_reader(uploaded_file):
    """
    Reads CSV with automatic delimiter detection.
    Handles edge cases like single-column comma strings.
    """
    content = uploaded_file.read().decode("utf-8", errors="replace")
    uploaded_file.seek(0)

    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(content[:2048], delimiters=",;\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    df = pd.read_csv(uploaded_file, delimiter=delimiter)

    if df.shape[1] == 1 and "," in str(df.iloc[0, 0]):
        header = df.iloc[0, 0].split(",")
        data = df.iloc[1, 0].split(",")
        df = pd.DataFrame([data], columns=header)

    df.columns = [c.strip() for c in df.columns]
    return df

