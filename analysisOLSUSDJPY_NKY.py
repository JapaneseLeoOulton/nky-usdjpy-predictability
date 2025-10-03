from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

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

def compute_log_returns(df: pd.DataFrame, price_col: str, new_col: str) -> pd.DataFrame:
    out = df.copy()
    out[new_col] = np.log(out[price_col] / out[price_col].shift(1))
    return out

def correlation_tests(ret_df: pd.DataFrame, x_col: str = "r_USDJPY", y_col: str = "r_NKY"):
    x = ret_df[x_col].dropna()
    y = ret_df[y_col].dropna()
    xy = pd.concat([x, y], axis=1).dropna()
    x, y = xy[x_col].values, xy[y_col].values
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "n_obs": len(x)
    }

def rolling_correlation(ret_df: pd.DataFrame, x_col: str = "r_USDJPY", y_col: str = "r_NKY", window: int = 60) -> pd.Series:
    return ret_df[x_col].rolling(window).corr(ret_df[y_col])

def ols_hac(ret_df: pd.DataFrame, x_col: str = "r_USDJPY", y_col: str = "r_NKY", maxlags: int = 5):
    xy = ret_df[[x_col, y_col]].dropna()
    X = sm.add_constant(xy[x_col].values)
    y = xy[y_col].values
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    beta = model.params[1]
    beta_se = model.bse[1]
    beta_t = model.tvalues[1]
    beta_p = model.pvalues[1]
    r2 = model.rsquared
    sx = xy[x_col].std(ddof=1)
    sy = xy[y_col].std(ddof=1)
    beta_std = beta * (sx / sy) if sy != 0 else np.nan
    eff_per_1pct_fx = beta * 0.01
    return {
        "beta": beta,
        "beta_se": beta_se,
        "beta_t": beta_t,
        "beta_p": beta_p,
        "r2": r2,
        "beta_std": beta_std,
        "eff_per_1pct_fx": eff_per_1pct_fx,
        "n_obs": len(xy)
    }

def ols_quadratic_hac(ret_df: pd.DataFrame, x_col: str = "r_USDJPY", y_col: str = "r_NKY", maxlags: int = 5):
    xy = ret_df[[x_col, y_col]].dropna().copy()
    xy["x2"] = xy[x_col] ** 2
    X = sm.add_constant(xy[[x_col, "x2"]].values)
    y = xy[y_col].values
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    params = model.params
    pvals = model.pvalues
    return {
        "beta_x": params[1], "p_x": pvals[1],
        "beta_x2": params[2], "p_x2": pvals[2],
        "r2": model.rsquared
    }

def quantile_regression(ret_df: pd.DataFrame, taus=(0.1, 0.5, 0.9), x_col: str = "r_USDJPY", y_col: str = "r_NKY"):
    xy = ret_df[[x_col, y_col]].dropna()
    X = sm.add_constant(xy[x_col].values)
    y = xy[y_col].values
    out = []
    for tau in taus:
        rq = sm.QuantReg(y, X).fit(q=tau)
        out.append({"tau": tau, "beta": rq.params[1]})
    return pd.DataFrame(out)

def binned_state_dependence(ret_df: pd.DataFrame, x_col="r_USDJPY", y_col="r_NKY"):
    q = ret_df[x_col].quantile([0.0, 0.1, 0.3, 0.7, 0.9, 1.0]).values
    bins = pd.IntervalIndex.from_tuples(list(zip(q[:-1], q[1:])), closed="right")
    lab = [f"{i.left:.3%}–{i.right:.3%}" for i in bins]
    cats = pd.cut(ret_df[x_col], bins=bins)
    g = ret_df.groupby(cats)[y_col].agg(["mean","median","count"]).reset_index(drop=True)
    g.insert(0, "USDJPY_bin", lab)
    return g

def main():
    nky = load_nikkei()
    fx = load_usdjpy()
    df = merge_align(nky, fx)

    if "NKY_Close" not in df.columns or "USDJPY_Price" not in df.columns:
        print("Close columns not found. Available columns:", df.columns.tolist())
        return

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twinx()
    ax1.plot(df.index, df["NKY_Close"], label="Nikkei Close (JPY)")
    ax2.plot(df.index, df["USDJPY_Price"], linestyle="--", label="USD/JPY")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Nikkei (JPY)")
    ax2.set_ylabel("USD/JPY (JPY per USD)")
    ax1.set_title("Nikkei vs USD/JPY Closing Prices")
    fig.tight_layout()
    fig.savefig("fig_prices.png")
    plt.close(fig)

    df = compute_log_returns(df, "NKY_Close", "r_NKY")
    df = compute_log_returns(df, "USDJPY_Price", "r_USDJPY")
    ret_df = df[["r_NKY", "r_USDJPY"]].dropna()

    cor_res = correlation_tests(ret_df, "r_USDJPY", "r_NKY")
    print("\n=== Correlation Tests (returns) ===")
    print(f"N obs: {cor_res['n_obs']}")
    print(f"Pearson r = {cor_res['pearson_r']:.4f}, p = {cor_res['pearson_p']:.4g}")
    print(f"Spearman ρ = {cor_res['spearman_rho']:.4f}, p = {cor_res['spearman_p']:.4g}")

    roll_win = 60
    roll_corr = rolling_correlation(ret_df, "r_USDJPY", "r_NKY", window=roll_win)
    fig2, ax = plt.subplots(figsize=(12,4))
    ax.plot(roll_corr.index, roll_corr, label=f"Rolling {roll_win}D Corr (NKY vs USDJPY)")
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.set_title(f"Rolling {roll_win}-Day Pearson Correlation: r(NKY returns, USDJPY returns)")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig("fig_rolling_corr.png")
    plt.close(fig2)

    ols_res = ols_hac(ret_df, "r_USDJPY", "r_NKY", maxlags=5)
    print("\n=== OLS with Newey–West (HAC) ===")
    print(f"N obs: {ols_res['n_obs']}")
    print(f"beta (USDJPY→NKY) = {ols_res['beta']:.6f}")
    print(f"SE_HAC = {ols_res['beta_se']:.6f}, t = {ols_res['beta_t']:.3f}, p = {ols_res['beta_p']:.4g}")
    print(f"R^2 = {ols_res['r2']:.4f}")
    print(f"Standardized beta = {ols_res['beta_std']:.4f}")
    print(f"Effect for +1% FX move ≈ {ols_res['eff_per_1pct_fx']:.6f} (~{ols_res['eff_per_1pct_fx']*100:.4f}% NKY)")

    quad = ols_quadratic_hac(ret_df, "r_USDJPY", "r_NKY", maxlags=5)
    print("\n=== Quadratic Term Check (HAC) ===")
    print(f"beta_x = {quad['beta_x']:.6f}, p_x = {quad['p_x']:.4g}")
    print(f"beta_x2 = {quad['beta_x2']:.6f}, p_x2 = {quad['p_x2']:.4g}")
    print(f"R^2 = {quad['r2']:.4f}")

    qr = quantile_regression(ret_df, taus=(0.1, 0.5, 0.9), x_col="r_USDJPY", y_col="r_NKY")
    print("\n=== Quantile Regression (beta by τ) ===")
    print(qr.to_string(index=False))

    bins = binned_state_dependence(ret_df, "r_USDJPY", "r_NKY")
    print("\n=== State Dependence by USDJPY-return bins ===")
    print(bins.to_string(index=False))

    print("\nSaved: fig_prices.png, fig_rolling_corr.png")

if __name__ == "__main__":
    main()
