import pandas as pd
from pathlib import Path

RAW_FILE    = Path("data/raw/nikkei.csv")
DATE_COL    ="Date"
VALUES_COLS  =["Open","High","Low","Close","Volume"]

def main():

    df = pd.read_csv(RAW_FILE)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).set_index(DATE_COL).sort_index()
    if VALUES_COLS:
        df = df[VALUES_COLS]

    print("Shape:", df.shape)
    print("Head:\n", df.head(5))
    print("Tail:\n", df.tail(5))
    print("Missing values:\n", df.isna().sum())

if __name__ == "__main__":
    main()
