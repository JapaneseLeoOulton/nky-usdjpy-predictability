import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/processed/merged.csv")
OUT_CSV = Path("data/processed/predict_results.csv")
OUT_JSON = Path("data/processed/predict_summary.json")
INITIAL_WINDOW_MIN = 500
TRADING_DAYS = 252
COST_BPS = 5
VAR_MAX_LAGS = 10

def nw_bandwidth(T):
    return max(1, int(4 * (T / 100) ** (2 / 9)))

def msfe(y, yhat):
    e = y - yhat
    return float(np.mean(np.square(e))), e

def oos_r2(y, y_model, y_bench):
    m_model, _ = msfe(y, y_model)
    m_bench, _ = msfe(y, y_bench)
    return 1.0 - m_model / m_bench, m_model, m_bench

def lr_var(d, L):
    d = np.asarray(d, dtype=float)
    d = d - d.mean()
    n = len(d)
    s = np.sum(d * d) / n
    for k in range(1, L + 1):
        gamma = np.sum(d[k:] * d[:-k]) / n
        w = 1.0 - k / (L + 1.0)
        s += 2.0 * w * gamma
    return s

def clark_west(y, f_full, f_nested, L):
    y = np.asarray(y, float)
    f_full = np.asarray(f_full, float)
    f_nested = np.asarray(f_nested, float)
    u_full = y - f_full
    u_nested = y - f_nested
    a = u_nested**2 - (u_full**2 - (f_full - f_nested)**2)
    abar = float(np.mean(a))
    s = lr_var(a, L)
    if s <= 0 or np.isnan(s):
        return np.nan, np.nan
    stat = abar / np.sqrt(s / len(a))
    from statistics import NormalDist
    p = 2 * (1 - NormalDist().cdf(abs(stat)))
    return float(stat), float(p)

def expanding_oos(y, X_const, init):
    preds = np.full(len(y), np.nan)
    for t in range(init, len(y)):
        Xt = X_const.iloc[:t]
        yt = y.iloc[:t]
        m = OLS(yt, Xt).fit()
        preds[t] = m.predict(X_const.iloc[t:t+1])[0]
    return pd.Series(preds, index=y.index)

def ar1_oos(y, init):
    ylag = y.shift(1)
    X = add_constant(ylag.to_frame(name="ylag"), has_constant="add")
    preds = np.full(len(y), np.nan)
    for t in range(init, len(y)):
        if pd.isna(X.iloc[t]["ylag"]):
            continue
        idx = X.index[:t]
        idx = idx.intersection(y.index[:t])
        idx = idx.intersection(X.dropna().index)
        if len(idx) == 0:
            continue
        m = OLS(y.loc[idx], X.loc[idx]).fit()
        preds[t] = m.predict(X.iloc[t:t+1])[0]
    return pd.Series(preds, index=y.index)

def perf_stats(r):
    mu = r.mean() * TRADING_DAYS
    sig = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = 0.0 if sig == 0 or np.isnan(sig) else mu / sig
    eq = (1 + r).cumprod()
    cagr = np.nan
    if len(eq) > 1 and eq.iloc[0] != 0 and not np.isnan(eq.iloc[-1]):
        years = len(r) / TRADING_DAYS
        if years > 0:
            cagr = eq.iloc[-1] ** (1 / years) - 1
    return float(mu), float(sig), float(sharpe), float(cagr)

def subperiod_oos_r2(dates, y, f_full, f_ctrl, start, split_date):
    mask = (dates >= start) & (dates < split_date)
    r2a, _, _ = oos_r2(y[mask], f_full[mask], f_ctrl[mask]) if mask.any() else (np.nan, np.nan, np.nan)
    mask2 = dates >= split_date
    r2b, _, _ = oos_r2(y[mask2], f_full[mask2], f_ctrl[mask2]) if mask2.any() else (np.nan, np.nan, np.nan)
    return float(r2a), float(r2b)

def main():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    for c in ["NKY", "SPX", "USDJPY", "VIX"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["NKY", "SPX", "USDJPY", "VIX"])
    if not (df.index.is_monotonic_increasing and df.index.is_unique):
        df = df[~df.index.duplicated(keep="first")].sort_index()

    df["r_NKY"] = np.log(df["NKY"]).diff()
    df["r_SPX_lag1"] = np.log(df["SPX"]).diff().shift(1)
    df["r_FX_lag1"] = np.log(df["USDJPY"]).diff().shift(1)
    df["dVIX_lag1"] = df["VIX"].diff().shift(1)
    df["VIX_lag1"] = df["VIX"].shift(1)

    dfm = df.dropna(subset=["r_NKY", "r_SPX_lag1", "r_FX_lag1", "dVIX_lag1", "VIX_lag1"]).copy()
    y = dfm["r_NKY"]
    Xc = dfm[["r_SPX_lag1", "dVIX_lag1", "VIX_lag1"]]
    Xf = dfm[["r_FX_lag1", "r_SPX_lag1", "dVIX_lag1", "VIX_lag1"]]

    Xc_const = add_constant(Xc, has_constant="add")
    Xf_const = add_constant(Xf, has_constant="add")

    T = len(dfm)
    L = nw_bandwidth(T)
    res_c = OLS(y, Xc_const).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    res_f = OLS(y, Xf_const).fit(cov_type="HAC", cov_kwds={"maxlags": L})

    r2_c = float(res_c.rsquared)
    r2_f = float(res_f.rsquared)
    d_r2 = r2_f - r2_c

    yhat_c_in = res_c.fittedvalues
    yhat_f_in = res_f.fittedvalues
    ssr_c = float(np.sum((y - yhat_c_in) ** 2))
    ssr_f = float(np.sum((y - yhat_f_in) ** 2))
    partial_r2 = (ssr_c - ssr_f) / ssr_c if ssr_c > 0 else np.nan

    if "r_FX_lag1" in res_f.params.index:
        t_fx = float(res_f.tvalues["r_FX_lag1"])
        p_fx = float(res_f.pvalues["r_FX_lag1"])
        beta_fx = float(res_f.params["r_FX_lag1"])
    else:
        t_fx, p_fx, beta_fx = np.nan, np.nan, np.nan

    IS_signal = (abs(t_fx) >= 1.96) and (d_r2 > 0)

    N = len(dfm)
    init = max(INITIAL_WINDOW_MIN, int(0.3 * N))
    init = min(init, N - 50)

    oos_f = expanding_oos(y, Xf_const, init)
    oos_c = expanding_oos(y, Xc_const, init)
    oos_ar = ar1_oos(y, init)

    mask = ~oos_f.isna()
    y_oos = y[mask]
    f_full = oos_f[mask]
    f_ctrl = oos_c[mask]
    f_ar = oos_ar[mask]
    dates_oos = y_oos.index

    r2_vs_ctrl, mse_full_ctrl, mse_ctrl = oos_r2(y_oos, f_full, f_ctrl)
    r2_vs_ar, mse_full_ar, mse_ar = oos_r2(y_oos, f_full, f_ar)

    cw_full_ctrl_stat, cw_full_ctrl_p = clark_west(y_oos.values, f_full.values, f_ctrl.values, nw_bandwidth(len(y_oos)))
    cw_full_ar_stat, cw_full_ar_p = clark_west(y_oos.values, f_full.values, f_ar.values, nw_bandwidth(len(y_oos)))

    OOS_signal = (r2_vs_ctrl > 0.0) and (cw_full_ctrl_p <= 0.10 if not np.isnan(cw_full_ctrl_p) else False)

    sign_full = np.sign(f_full)
    sign_ctrl = np.sign(f_ctrl)
    sign_ar = np.sign(f_ar)
    acc_full = float((np.sign(y_oos) == sign_full).mean())
    acc_ctrl = float((np.sign(y_oos) == sign_ctrl).mean())
    acc_ar = float((np.sign(y_oos) == sign_ar).mean())

    pos = (f_full > 0).astype(int)
    pos_lag = pos.shift(1).fillna(0).astype(int)
    trade_switch = (pos != pos_lag).astype(int)
    cost = trade_switch * (COST_BPS / 10000.0)
    strat_r = pos_lag * y_oos - cost.reindex(y_oos.index).fillna(0.0)

    mu_s, sig_s, sh_s, cagr_s = perf_stats(strat_r)
    mu_b, sig_b, sh_b, cagr_b = perf_stats(y_oos)
    turnover = float(trade_switch.mean() * TRADING_DAYS)

    ECON_signal = (sh_s > 0.0) and (cagr_s is not None and cagr_s > 0.0) and (turnover <= 1.5)

    r2_pre2020, r2_post2020 = subperiod_oos_r2(dates_oos, y_oos, f_full, f_ctrl, pd.Timestamp("1900-01-01"), pd.Timestamp("2020-01-01"))
    r2_pre202408, r2_post202408 = subperiod_oos_r2(dates_oos, y_oos, f_full, f_ctrl, pd.Timestamp("1900-01-01"), pd.Timestamp("2024-08-01"))
    stable_sign = (np.sign(r2_pre2020) == np.sign(r2_post2020) or np.isnan(r2_pre2020) or np.isnan(r2_post2020)) and (np.sign(r2_pre202408) == np.sign(r2_post202408) or np.isnan(r2_pre202408) or np.isnan(r2_post202408))
    STABLE_signal = stable_sign and not (r2_post2020 < -0.01 or r2_post202408 < -0.01)

    try:
        df_var = pd.DataFrame({
            "NKY": df["r_NKY"],
            "USDJPY": np.log(df["USDJPY"]).diff(),
            "SPX": np.log(df["SPX"]).diff(),
            "dVIX": df["VIX"].diff()
        }).dropna()
        var = VAR(df_var)
        sel = var.select_order(VAR_MAX_LAGS)
        p = sel.selected_orders.get("aic") or sel.selected_orders.get("bic") or 1
        p = int(max(1, min(VAR_MAX_LAGS, p)))
        var_res = var.fit(p)
        gc = var_res.test_causality(caused="NKY", causing=["USDJPY"], kind="f")
        var_lags = int(var_res.k_ar)
        var_f = float(gc.test_statistic)
        var_p = float(gc.pvalue)
    except Exception:
        var_lags, var_f, var_p = np.nan, np.nan, np.nan

    PIDX = 0.0
    PIDX += 30.0 * min(1.0, max(0.0, 10.0 * max(0.0, d_r2)))
    PIDX += 40.0 * max(0.0, r2_vs_ctrl)
    PIDX += 20.0 * (1.0 if (not np.isnan(cw_full_ctrl_p) and cw_full_ctrl_p <= 0.10) else 0.0)
    PIDX += 10.0 * (1.0 if (sh_s > 0.0 and (cagr_s is not None and cagr_s > 0.0)) else 0.0)
    PIDX = float(np.clip(PIDX, 0.0, 100.0))

    Proceed_to_ML = (OOS_signal or PIDX >= 35.0) and STABLE_signal

    out = pd.DataFrame({
        "y": y_oos,
        "pred_full": f_full,
        "pred_controls": f_ctrl,
        "pred_ar1": f_ar,
        "position_full_lag": pos_lag.reindex(y_oos.index).fillna(0).astype(int),
        "trade_switch": trade_switch.reindex(y_oos.index).fillna(0).astype(int),
        "cost": cost.reindex(y_oos.index).fillna(0.0),
        "ret_strategy_after_cost": strat_r
    })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV)

    summary = {
        "sample": {
            "start": str(dfm.index[0].date()) if len(dfm) else None,
            "end": str(dfm.index[-1].date()) if len(dfm) else None,
            "n_obs_model": int(len(dfm)),
            "n_obs_oos": int(len(y_oos))
        },
        "in_sample": {
            "beta_fx": beta_fx,
            "t_fx_hac": t_fx,
            "p_fx_hac": p_fx,
            "r2_controls": r2_c,
            "r2_full": r2_f,
            "delta_r2": d_r2,
            "partial_r2_fx_cond_controls": float(partial_r2),
            "hac_maxlags": int(L),
            "signal_flag": bool(IS_signal)
        },
        "oos": {
            "r2_vs_controls": r2_vs_ctrl,
            "r2_vs_ar1": r2_vs_ar,
            "cw_full_vs_controls_stat": cw_full_ctrl_stat,
            "cw_full_vs_controls_p": cw_full_ctrl_p,
            "cw_full_vs_ar1_stat": cw_full_ar_stat,
            "cw_full_vs_ar1_p": cw_full_ar_p,
            "acc_full": acc_full,
            "acc_controls": acc_ctrl,
            "acc_ar1": acc_ar,
            "signal_flag": bool(OOS_signal)
        },
        "economic": {
            "cost_bps_per_switch": COST_BPS,
            "annual_turnover": turnover,
            "strategy_ann_return": mu_s,
            "strategy_ann_vol": sig_s,
            "strategy_sharpe": sh_s,
            "strategy_cagr": cagr_s,
            "buyhold_ann_return": mu_b,
            "buyhold_ann_vol": sig_b,
            "buyhold_sharpe": sh_b,
            "buyhold_cagr": cagr_b,
            "signal_flag": bool(ECON_signal)
        },
        "stability": {
            "r2_pre2020_vs_controls": r2_pre2020,
            "r2_post2020_vs_controls": r2_post2020,
            "r2_pre2024Aug_vs_controls": r2_pre202408,
            "r2_post2024Aug_vs_controls": r2_post202408,
            "signal_flag": bool(STABLE_signal)
        },
        "var_granger": {
            "lags_used": var_lags,
            "f_stat": var_f,
            "p_value": var_p
        },
        "index": {
            "PIDX": PIDX,
            "Proceed_to_ML": bool(Proceed_to_ML)
        },
        "artifacts": {
            "predictions_csv": str(OUT_CSV.resolve()),
            "summary_json": str(OUT_JSON.resolve())
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Predictability Summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
