from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NIKKEI_CSV = Path("data/raw/nikkei.csv")
USDJPY_CSV = Path("data/raw/usdjpy.csv")   
def load_nikkei(path=NIKKEI_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    df = df[keep].add_prefix("NKY_")
    return df

def load_usdjpy(path=USDJPY_CSV) -> pd.DataFrame:
    fx = pd.read_csv(path)
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=False, errors="coerce")
    fx = fx.dropna(subset=["Date"]).set_index("Date").sort_index()
    keep = [c for c in ["Open","High","Low","Price","Volume"] if c in fx.columns]
    fx = fx[keep].add_prefix("USDJPY_")
    return fx
def merge_align(nky: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
     df = nky.join(fx, how="inner")
     return df.dropna()

def main():
    nky = load_nikkei()
    fx = load_usdjpy()
    df = merge_align(nky, fx)
    if "NKY_Close" not in df.columns or "USDJPY_Price" not in df.columns:
        print("Close columns not found. Available columns:", df.columns.tolist())
        return
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["NKY_Close"], color="tab:blue", label="Nikkei Close (JPY)")
    ax2.plot(df.index, df["USDJPY_Price"], color="tab:red", linestyle="--", label="USD/JPY")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Nikkei (JPY)", color="tab:blue")
    ax2.set_ylabel("USD/JPY (JPY per USD)", color="tab:red")
    ax1.set_title("Nikkei vs USD/JPY Closing Prices")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
