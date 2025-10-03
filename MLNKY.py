import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, norm
import statsmodels.api as sm
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

MERGED_CSV = Path("data/processed/merged.csv")
RESULTS_DIR = Path("results")
TRAIN_START = pd.Timestamp("2023-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-08-31")
RANDOM_STATE = 42

def safe_spearman(x, y):
    x = pd.Series(x)
    y = pd.Series(y)
    if x.size == 0 or y.size == 0:
        return np.nan
    if np.nanstd(x.values) == 0 or np.nanstd(y.values) == 0:
        return np.nan
    return spearmanr(x, y).correlation

def ensure_results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def read_merged(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Merged CSV not found at {path}")
    df = pd.read_csv(path)
    date_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ["date"] or "date" in cl or "time" in cl:
            date_col = c
            break
    if date_col is None:
        raise ValueError("No date-like column found in merged CSV.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    return df

def find_col(cols, keywords):
    lc = {c.lower(): c for c in cols}
    for c in cols:
        cl = c.lower()
        ok = True
        for k in keywords:
            if k not in cl:
                ok = False
                break
        if ok:
            return c
    for k in keywords:
        if k in lc:
            return lc[k]
    return None

def first_available(df, candidates):
    for kws in candidates:
        col = find_col(df.columns, kws)
        if col is not None:
            return col
    return None

def logret(s):
    return np.log(s/s.shift(1))

def build_dataset(df):
    nky_close = first_available(df, [["nky","close"],["nikkei","close"],["ni225","close"],["^n225","close"],["nky_close"],["close"]])
    fx_close = first_available(df, [["usdjpy","close"],["fx","close"],["jpy","close"],["usd/jpy","close"],["usdjpy_close"],["usdjpy"]])
    spx_close = first_available(df, [["spx","close"],["^gspc","close"],["sp500","close"],["spx_close"]])
    vix_col = first_available(df, [["vix"],["vix","close"],["^vix"]])
    if nky_close is None or fx_close is None:
        raise ValueError("Required columns not found. Expecting NKY close and USDJPY close in merged CSV.")
    price = df[nky_close].astype(float)
    r_nky = logret(price)
    fx = df[fx_close].astype(float)
    r_fx = logret(fx)
    feats = pd.DataFrame(index=df.index)
    for k in range(1,6):
        feats[f"r_fx_lag{k}"] = r_fx.shift(k)
    for k in range(1,4):
        feats[f"r_nky_lag{k}"] = r_nky.shift(k)
    feats["fx_vol5"] = r_fx.rolling(5).std()
    feats["fx_vol20"] = r_fx.rolling(20).std()
    if spx_close is not None:
        spx = df[spx_close].astype(float)
        r_spx = logret(spx)
        feats["r_spx_lag1"] = r_spx.shift(1)
    if vix_col is not None:
        vix = df[vix_col].astype(float)
        feats["dvix_lag1"] = vix.diff().shift(1)
        feats["vix_lag1"] = vix.shift(1)
    dows = pd.get_dummies(feats.index.dayofweek, prefix="dow", drop_first=True)
    dows.index = feats.index
    feats = pd.concat([feats, dows], axis=1)
    y = r_nky.shift(-1)
    data = feats.copy()
    data["y"] = y
    data["price"] = price
    data["r_nky"] = r_nky
    return data.dropna()

def slice_period(data, start, end):
    return data.loc[(data.index>=start)&(data.index<=end)].copy()

def gen_splits(n_splits, X):
    tss = TimeSeriesSplit(n_splits=n_splits)
    return list(tss.split(X))

def eval_cv(model, X, y, splits):
    mses = []
    ics = []
    for tr, va in splits:
        Xt = X.iloc[tr]
        yt = y.iloc[tr]
        Xv = X.iloc[va]
        yv = y.iloc[va]
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(Xt, yt)
        pv = pipe.predict(Xv)
        mses.append(mean_squared_error(yv, pv))
        try:
            ic = safe_spearman(pv, yv)
        except Exception:
            ic = np.nan
        ics.append(ic)
    return np.nanmean(mses), np.nanmean(ics)

def select_model(X, y, splits):
    candidates = []
    for a in [0.1,1.0,10.0]:
        candidates.append(("Ridge", Ridge(alpha=a, random_state=42)))
    for a in [0.0005,0.001,0.005]:
        candidates.append(("Lasso", Lasso(alpha=a, random_state=42, max_iter=20000)))
    for a in [0.001,0.01]:
        for l1 in [0.2,0.5,0.8]:
            candidates.append(("ElasticNet", ElasticNet(alpha=a, l1_ratio=l1, random_state=42, max_iter=20000)))
    for lr in [0.03,0.05]:
        for md in [2,3]:
            for ss in [0.7,1.0]:
                candidates.append(("GBR", GradientBoostingRegressor(random_state=42, n_estimators=400, learning_rate=lr, max_depth=md, subsample=ss)))
    if HAS_XGB:
        for lr in [0.03,0.05]:
            for md in [2,3]:
                for mcw in [2,5]:
                    candidates.append(("XGB", XGBRegressor(n_estimators=600, learning_rate=lr, max_depth=md, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=mcw, objective="reg:squarederror", random_state=42, n_jobs=2)))
    records = []
    for name, mdl in candidates:
        mse, ic = eval_cv(mdl, X, y, splits)
        records.append((name, mdl.get_params(), mse, ic))
    df = pd.DataFrame(records, columns=["model","params","cv_mse","cv_ic"]).sort_values(["cv_mse","cv_ic"], ascending=[True,False]).reset_index(drop=True)
    return df

def fit_best(model_name, params, X, y):
    if model_name=="Ridge":
        model = Ridge(**params)
    elif model_name=="Lasso":
        model = Lasso(**params)
    elif model_name=="ElasticNet":
        model = ElasticNet(**params)
    elif model_name=="GBR":
        model = GradientBoostingRegressor(**params)
    elif model_name=="XGB" and HAS_XGB:
        model = XGBRegressor(**params)
    else:
        model = Ridge(alpha=1.0, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X, y)
    return pipe

def dm_test(loss_model, loss_base, lag=5):
    d = loss_model - loss_base
    d = d - np.mean(d)
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    gamma0 = np.dot(d, d)/T
    var = gamma0
    for k in range(1, min(lag, T-1)+1):
        w = 1 - k/(lag+1)
        cov = np.dot(d[k:], d[:-k])/T
        var += 2*w*cov
    var_mean = var / T
    if var_mean<=0:
        return np.nan, np.nan
    dm = np.mean(loss_model - loss_base) / np.sqrt(var_mean)
    p = 2*(1 - norm.cdf(abs(dm)))
    return dm, p

def mz_regression(y_true, y_pred):
    X = pd.DataFrame({"const": 1.0, "yhat": pd.Series(y_pred, index=y_true.index)})
    model = sm.OLS(y_true, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    a = model.params["const"]
    b = model.params["yhat"]
    pa = model.pvalues["const"]
    pb = model.pvalues["yhat"]
    return a, b, pa, pb


def ar1_baseline(train_ret, test_ret):
    df = pd.DataFrame({"y": train_ret.shift(-1), "x": train_ret}).dropna()
    if len(df)<10:
        return pd.Series(0.0, index=test_ret.index)
    X = sm.add_constant(df["x"])
    mdl = sm.OLS(df["y"], X).fit()
    x_test = sm.add_constant(test_ret, has_constant="add")
    pred = mdl.predict(x_test)
    pred = pred.reindex(test_ret.index).fillna(0.0)
    return pred

def controls_only_baseline(train_df, test_df):
    cols = [c for c in ["r_spx_lag1","dvix_lag1","vix_lag1"] if c in train_df.columns]
    if len(cols)==0:
        return pd.Series(0.0, index=test_df.index)
    dtr = train_df.dropna(subset=cols+["y"]).copy()
    if len(dtr)<30:
        return pd.Series(0.0, index=test_df.index)
    X = sm.add_constant(dtr[cols])
    mdl = sm.OLS(dtr["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags":1})
    Xt = sm.add_constant(test_df[cols], has_constant="add")
    pred = pd.Series(mdl.predict(Xt), index=test_df.index)
    pred = pred.reindex(test_df.index).fillna(0.0)
    return pred

def rolling_spearman(a, b, window=20):
    vals = []
    idx = a.index
    for i in range(len(a)):
        if i+1 < window:
            vals.append(np.nan)
        else:
            x = a.iloc[i+1-window:i+1]
            y = b.iloc[i+1-window:i+1]
            vals.append(spearmanr(x, y).correlation if x.notna().all() and y.notna().all() else np.nan)
    return pd.Series(vals, index=idx)

def rolling_hitrate(y_true, y_pred, window=20):
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    hit = (s_true == s_pred).astype(float)
    return hit.rolling(window).mean()

def plot_returns(y_true, y_pred, fname):
    plt.figure(figsize=(10,4))
    plt.plot(y_true.index, y_true.values, label="Actual Return")
    plt.plot(y_pred.index, y_pred.values, label="Predicted Return")
    plt.title("NKY Returns: Actual vs Predicted (2025-01-01 to 2025-08-31)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()

def plot_scatter_mz(y_true, y_pred, a, b, fname):
    plt.figure(figsize=(5,5))
    plt.scatter(y_pred.values, y_true.values, s=12)
    x = np.linspace(y_pred.min(), y_pred.max(), 100)
    y = a + b*x
    plt.plot(x, y)
    plt.title("Mincer–Zarnowitz: y vs y_hat (2025-01-01 to 2025-08-31)")
    plt.xlabel("Predicted Return")
    plt.ylabel("Actual Return")
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()

def plot_price_vs_implied(price, y_pred, fname):
    start_price = float(price.iloc[0])
    implied = start_price * np.exp(pd.Series(y_pred.values, index=y_pred.index).cumsum())
    plt.figure(figsize=(10,4))
    plt.plot(price.index, price.values, label="Actual NKY")
    plt.plot(implied.index, implied.values, label="Predicted-Implied")
    plt.title("NKY Price vs Implied Predicted Path (2025-01-01 to 2025-08-31)")
    plt.xlabel("Date")
    plt.ylabel("Index Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()
    return implied

def plot_rolling_metrics(ic_series, hr_series, ic_fname, hr_fname):
    plt.figure(figsize=(10,3.5))
    plt.plot(ic_series.index, ic_series.values)
    plt.title("Rolling 20D Spearman IC (2025-01-01 to 2025-08-31)")
    plt.xlabel("Date")
    plt.ylabel("IC")
    plt.tight_layout()
    plt.savefig(ic_fname, dpi=160)
    plt.close()
    plt.figure(figsize=(10,3.5))
    plt.plot(hr_series.index, hr_series.values)
    plt.title("Rolling 20D Hit-Rate (2025-01-01 to 2025-08-31)")
    plt.xlabel("Date")
    plt.ylabel("Hit-Rate")
    plt.tight_layout()
    plt.savefig(hr_fname, dpi=160)
    plt.close()

def main():
    ensure_results_dir()
    path = MERGED_CSV if len(sys.argv)<2 else Path(sys.argv[1])
    df = read_merged(path)
    data = build_dataset(df)
    data = data.loc[data.index>=TRAIN_START]
    trainval = slice_period(data, TRAIN_START, TRAIN_END)
    test = slice_period(data, TEST_START, TEST_END)
    if len(test)==0:
        test = data.loc[data.index>=TEST_START]
    X_cols = [c for c in data.columns if c not in ["y","price","r_nky"]]
    X_trv = trainval[X_cols]
    y_trv = trainval["y"]
    splits = gen_splits(4, X_trv)
    cv_table = select_model(X_trv, y_trv, splits)
    cv_table.to_json(RESULTS_DIR/"cv_summary.json", orient="records", indent=2, date_format="iso")
    best_row = cv_table.iloc[0]
    best_name = best_row["model"]
    best_params = best_row["params"]
    pipe = fit_best(best_name, best_params, X_trv, y_trv)
    X_te = test[X_cols]
    y_te = test["y"]
    yhat = pd.Series(pipe.predict(X_te), index=y_te.index, name="yhat")
    y_true = y_te.copy()
    mse = mean_squared_error(y_true, yhat)
    mae = mean_absolute_error(y_true, yhat)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, yhat)
    ic = ic = safe_spearman(yhat, y_true)
    a, b, pa, pb = mz_regression(y_true, yhat)
    rw_pred = pd.Series(0.0, index=y_true.index, name="rw")
    ar_pred = ar1_baseline(trainval["r_nky"], test["r_nky"])
    co_pred = controls_only_baseline(trainval, test)
    loss_model = (y_true - yhat)**2
    loss_rw = (y_true - rw_pred)**2
    loss_ar = (y_true - ar_pred)**2
    loss_co = (y_true - co_pred)**2
    dm_rw, p_rw = dm_test(loss_model.values, loss_rw.values, lag=5)
    dm_ar, p_ar = dm_test(loss_model.values, loss_ar.values, lag=5)
    dm_co, p_co = dm_test(loss_model.values, loss_co.values, lag=5)
    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "ic_spearman": ic,
        "mz_alpha": a,
        "mz_beta": b,
        "mz_p_alpha": pa,
        "mz_p_beta": pb,
        "dm_vs_randomwalk": dm_rw,
        "p_dm_vs_randomwalk": p_rw,
        "dm_vs_ar1": dm_ar,
        "p_dm_vs_ar1": p_ar,
        "dm_vs_controls_only": dm_co,
        "p_dm_vs_controls_only": p_co,
        "test_start": str(y_true.index.min().date() if len(y_true)>0 else ""),
        "test_end": str(y_true.index.max().date() if len(y_true)>0 else ""),
        "n_test": int(len(y_true))
    }
    pd.Series(metrics).to_csv(RESULTS_DIR/"metrics_test_2025Aug.csv")
    preds = pd.DataFrame({"y_true": y_true, "yhat": yhat, "rw": rw_pred, "ar1": ar_pred, "controls_only": co_pred}, index=y_true.index)
    preds.to_csv(RESULTS_DIR/"predictions_2025Aug.csv")
    plot_returns(y_true, yhat, RESULTS_DIR/"plot_returns_timeseries_2025Aug.png")
    plot_scatter_mz(y_true, yhat, a, b, RESULTS_DIR/"plot_returns_scatter_MZ_2025Aug.png")
    price_test = test["price"].reindex(y_true.index).ffill()
    implied = plot_price_vs_implied(price_test, yhat, RESULTS_DIR/"plot_price_vs_pred_implied_2025Aug.png")
    ic_roll = rolling_spearman(y_true, yhat, window=20)
    hr_roll = rolling_hitrate(y_true, yhat, window=20)
    plot_rolling_metrics(ic_roll, hr_roll, RESULTS_DIR/"plot_rolling_IC_2025Aug.png", RESULTS_DIR/"plot_rolling_hitrate_2025Aug.png")
    summary_text = [
        f"Model: {best_name}",
        f"MSE: {mse:.6e}",
        f"MAE: {mae:.6e}",
        f"RMSE: {rmse:.6e}",
        f"R2: {r2:.4f}",
        f"IC (Spearman): {ic:.4f}",
        f"MZ alpha: {a:.4e} (p={pa:.3f})",
        f"MZ beta: {b:.3f} (p={pb:.3f})",
        f"DM vs RW: {dm_rw:.3f} (p={p_rw:.3f})",
        f"DM vs AR1: {dm_ar:.3f} (p={p_ar:.3f})",
        f"DM vs Controls: {dm_co:.3f} (p={p_co:.3f})",
        f"Test window: {metrics['test_start']} → {metrics['test_end']}, N={metrics['n_test']}"
    ]
    with open(RESULTS_DIR/"summary_test_2025Aug.txt","w",encoding="utf-8") as f:
        f.write("\n".join(summary_text))
    print("\n".join(summary_text))

if __name__ == "__main__":
    main()
