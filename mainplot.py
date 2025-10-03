import pandas as pd
import matplotlib.pyplot as plt
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

    df.plot(subplots=True, figsize=(12, 8), title="Nikkei 225 Time Series")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
