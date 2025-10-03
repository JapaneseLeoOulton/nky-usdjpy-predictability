# 📈 Forecasting Nikkei 225 Returns with USD/JPY & Volatility Controls  

## Overview  
This project investigates the **forecastability of next-day Nikkei 225 (NKY) log returns** using **FX (USD/JPY)** and **volatility-related controls (SPX, VIX, ΔVIX)**.  
The goal was to rigorously test whether macro-financial variables improve predictability compared to baseline models (Random Walk, AR(1), controls-only regression).  

The project was conducted as a **mini quant research case study**, combining statistical econometrics with a systematic forecasting pipeline.  

---

## Methodology  

### Data Construction  
- **Daily data**: Nikkei 225 closes, USD/JPY closes, SPX, and VIX.  
- **Log returns**:  
  \[
  r_{t} = \ln \left(\frac{P_{t}}{P_{t-1}}\right)
  \]  
- Constructed USDJPY returns and ΔVIX analogously.  

### Econometric Models  
1. **OLS with HAC (Newey–West)**  
   - Forecast regression:  
     \[
     r^{NKY}_{t+1} = \alpha + \beta_{FX} r^{USDJPY}_{t} + \gamma_1 r^{SPX}_{t} + \gamma_2 \Delta VIX_{t} + \gamma_3 VIX_{t} + \epsilon_t
     \]  
   - HAC (heteroskedasticity & autocorrelation consistent) standard errors applied.  

2. **Forecast Evaluation**  
   - **Diebold–Mariano Test**: compared forecast errors against Random Walk, AR(1), and controls-only models.  
   - **Mincer–Zarnowitz Regression**:  
     \[
     r_{t+1} = \alpha + \beta \hat{r}_{t+1} + \epsilon_t
     \]  
     Tested unbiasedness (α = 0, β = 1).  

3. **Predictive R²**  
   - Measured in-sample and out-of-sample forecastability.  

---

## Challenges & Solutions  

- **Multicollinearity (SPX, VIX, ΔVIX highly correlated)**  
  → Addressed via careful control design and Elastic Net experiments.  

- **Autocorrelation in residuals**  
  → Corrected using Newey–West HAC covariance estimators.  

- **Non-stationarity**  
  → Differenced all price data into returns before regression.  

- **Low signal-to-noise ratio**  
  → Out-of-sample R² close to zero, demonstrating difficulty of outperforming Random Walk. Applied robust statistical testing to avoid false positives.  

---

## Results  

- **In-sample OLS (HAC)**  
  - Controls (SPX, ΔVIX, VIX) significant.  
  - USDJPY coefficient ≈ 0.15 (t ≈ 0.9, p ≈ 0.36) → weak incremental effect.  
  - R² ≈ 0.18 (controls-only).  

- **Out-of-sample (2025 test set)**  
  - Forecast errors not significantly better than Random Walk or AR(1).  
  - DM test p-values > 0.3 → no robust predictive edge.  

- **Conclusion**  
  Market efficiency largely holds — FX adds little incremental predictability.  
  However, the project demonstrated **rigorous quantitative methodology** and systematic forecast evaluation.  

---

## Tools Used  
- **Python**: pandas, NumPy, statsmodels, scikit-learn  
- **Econometrics**: OLS, HAC, Diebold–Mariano, Mincer–Zarnowitz  
- **Visualization**: matplotlib, seaborn  
