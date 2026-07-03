<div align="center">
  <h3>Comparison of Volatility Forecasts from Machine Learning Models and Conditional Heteroskedasticity Models</h3>
</div>
<div align="right">
  <h4>Denis Makarov</h4>
</div>
<div align="center">
  <h4>(04/06/2026)</h4>
  <h4>DQuant</h4>
  <h4>Keywords: machine learning, volatility, conditional heteroskedasticity, volatility forecast</h4>
</div>

---

## 1 Abstract

The paper compares the accuracy of volatility forecasts of GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) models and XGBoost, LightGBM machine learning models. Machine learning models are taken from the dquant library, and conditional heteroskedasticity from the arch library. Data is taken from 01.01.2014 to 31.12.2024 from the yfinance library. Tickers: BTC-USD (bitcoin/dollar), ETH-USD (ether/dollar), EURUSD=X (euro/dollar), BZ=F (oil), GC=F (gold), SI=F (silver), SPY. Training was carried out on data from 01.01.2014 to 01.01.2023, validation from 01.01.2023 to 31.12.2024. Quality metrics of forecasts: QLIKE, MAE. The Diebold-Mariano test is used to check the statistical significance of the results. Machine learning models showed a statistically significant advantage on all assets in terms of MAE and on 4 out of 7 assets by QLIKE.

## 2 Introduction

Volatility is a statistical financial indicator that characterizes the variability of the price of a financial instrument. Volatility in finance is usually defined as the standard deviation or variance of the return of an asset over a certain period of time.

Quantifying the degree of price variability is important in trading, volatility is used everywhere: from risk management to use in trading strategies based on volatility forecasting.

A revolutionary breakthrough in modeling volatility was made by Robert Engle in 1982, when he proposed the ARCH (Autoregressive Conditional Heteroskedasticity) model. In the ARCH model, the conditional variance at the current time is represented as a linear function of the squares of past values of the process.

Mathematically, the ARCH(q) model can be represented as follows:

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2$$

where:

- $\sigma_t^2$ — conditional variance at time $t$;
- $\omega$ — constant (usually positive);
- $\alpha_i$ — model parameters, which must be positive to ensure that the variance is positive;
- $\varepsilon_{t-i}$ — innovations of the process (usually assumed to be independent and identically distributed with zero mean).

For the development of the ARCH model, Robert Engle was awarded the Nobel Prize in Economics in 2003, which highlights the significance of this contribution to financial econometrics.

In 1986, Tim Bollerslev proposed a generalization of the ARCH model, which became known as GARCH (Generalized Autoregressive Conditional Heteroskedasticity). The key idea was to include past conditional variances in the equation for the current conditional variance:

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2$$

where $\beta_j$ — parameters reflecting the influence of past conditional variances.

This formulation turned out to be more parsimonious in terms of parameters and often the GARCH(1,1) model was sufficient to adequately describe volatility, whereas an equivalent ARCH model would require an infinite number of parameters.

Machine learning models allow you to process a huge amount of data, they are used in trading, including in predicting volatility. For this, in most cases, boosting machine learning models are used, they do not require as much data as neural networks, but at the same time their forecasts do not lag behind the forecasts of neural network models.

The purpose of this work is to compare the accuracy of 1-step-ahead volatility forecasts on 7 different assets using GARCH family models and gradient boosting models. GARCH models will be taken from the arch library, gradient boosting models from the author's library dquant. The accuracy of forecasts will be compared on the MAE and QLIKE metrics. The statistical significance of the most accurate models will be checked on the Diebold-Mariano test.

## 3 Methodology

### 3.1 Data

The data is taken from the python library yfinance. For training, a period of 9 years of OHLC data from the daily time frame was taken. For validation, 2 years. The following assets are used: BTC-USD (bitcoin/dollar), ETH-USD (ether/dollar), EURUSD=X (euro/dollar), BZ=F (oil), GC=F (gold), SI=F (silver), SPY.

It is important to use different markets with different price movements. For example, if you already have a euro/dollar pair, you do not need to use a pound/dollar pair. Due to the strong correlation between these two instruments, the results will be similar.

After obtaining the data, it is necessary to divide it into training and validation data. Our models will be trained on the training data. The validation data will only be used after training. The python code is provided:

```python
START_DATE = '2014-01-01'
SPLIT_DATE = '2023-01-01'
END_DATE = '2024-12-31'

def get_data(ticker, start_date, end_date):
    print(f"Loading {ticker} from {start_date} to {end_date}...")
    raw = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df = pd.DataFrame({
        'open':   raw[('Open', ticker)].values,
        'high':   raw[('High', ticker)].values,
        'low':    raw[('Low', ticker)].values,
        'close':  raw[('Close', ticker)].values,
        'volume': raw[('Volume', ticker)].values
    }, index=raw.index)
    return df

df = get_data('BTC-USD', START_DATE, END_DATE)
train_mask = df.index < SPLIT_DATE
test_mask = df.index >= SPLIT_DATE

df_train = df[train_mask].copy()
df_test = df[test_mask].copy()
```

### 3.2 Models used, their training and validation

#### 3.2.1 Machine learning models

XGBoost and LightGBM machine learning models from the DQuant python library will be used. This library provides a convenient interface for training ML models for volatility prediction, and it was created specifically for this purpose.

We will use returns as input data and Parkinson volatility as targets. The Parkinson volatility formula is as follows:

$$\sigma_p = \sqrt{\frac{1}{4n \ln 2} \sum_{i=1}^{n} \left(\ln \frac{H_i}{L_i}\right)^2}$$

The library provides a convenient interface for creating features and targets, but for clarity, we implement these functions manually. The library allows you to embed your own functions for creating features and targets.

```python
def parkinson_func(df):
    df = df.copy()
    df = df.iloc[1:]
    return np.array(np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']))**2
    ))

def return_func(df):
    return np.array(df['close'].pct_change().dropna())
```

We embed these functions into the model training code:

```python
from dquant.models import VolClustXGB, VolClustLightGBM

def dquant_xgb_train():
    model_dq = VolClustXGB({}, early_stopping=True, output=False, loss='QLIKE')
    model_dq.fit(
        df_train,
        feature_list=FEATURES,
        input_bars=INPUT_BARS,
        horizon=HORIZON,
        trees_count=TREES_COUNT,
        show_results=False,
        feature_func=return_func,
        target_func=parkinson_func
    )
    return model_dq
```

The training will be based on the QLIKE loss function, which was created specifically for training to predict volatility and MAE.

Training was conducted with the default hyperparameters. For XGBoost, these are:

```python
{
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 5,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cpu'
}
```

For LightGBM:

```python
{
    'learning_rate': 0.1,
    'max_depth': 6,
    'num_leaves': 2**6 - 1,
    'min_child_samples': 5,
    'min_split_gain': 0.1,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbosity': -1,
    'boosting_type': 'gbdt'
}
```

The optimal number of trees is selected during the training process.

Code for obtaining model predictions:

```python
def dquant_xgb_forecasting():
    model_dq = dquant_xgb_train()
    dquant_preds = []
    dquant_dates = []

    for i in range(INPUT_BARS, len(df_test)):
        test_date = df_test.index[i]

        available_data = df.loc[:test_date].iloc[-INPUT_BARS:].copy()
        if len(available_data) < INPUT_BARS:
            continue

        try:
            forecast_vals = model_dq.forecast(available_data, show=False)
            pred_vol = forecast_vals[-1] if HORIZON > 1 else forecast_vals[0]
        except Exception as e:
            print(f"DQuant forecast error on {test_date.date()}: {e}")
            continue
        actual_vol = df_test.loc[test_date, 'parkinson_vol']

        dquant_dates.append(test_date)
        dquant_preds.append({
            'date': test_date,
            'actual': actual_vol,
            'predicted': pred_vol
        })

    df_dquant = pd.DataFrame(dquant_preds).set_index('date')
    return df_dquant
```

The same should be done with VolClustLightGBM.

#### 3.2.2 Conditional Heteroskedasticity Models

Conditional heteroskedasticity models will be used: GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) from the python library arch.

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) – conditional heteroskedasticity model.

The formula looks like this:

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Where:

- $\sigma_t^2$ — conditional variance at time $t$;
- $\varepsilon_{t-1}$ — standardized residual at time $t-1$;
- $\omega$ — constant;
- $\beta$ — persistence coefficient;
- $\alpha$ — impact of the shock size.

Implementation in python:

```python
def garch_forecasting():
    garch_preds = []

    returns_full = df['returns'].dropna() * 100   # scaling for GARCH
    for test_date in df_test.index:
        hist_returns = returns_full[:test_date].iloc[:-1]
        if len(hist_returns) < 500:
            continue

        try:
            model_garch = arch_model(hist_returns, vol='GARCH', p=1, q=1, dist='normal')
            fitted = model_garch.fit(disp='off')
            forec = fitted.forecast(horizon=HORIZON, reindex=False)
            cond_var = forec.variance.values[-1, 0]
            pred_vol = np.sqrt(cond_var) / 100.0   # reverse scaling
        except Exception as e:
            print(f"GARCH forecast error on {test_date.date()}: {e}")
            continue

        actual_vol = df.loc[test_date, 'parkinson_vol']
        garch_preds.append({
            'date': test_date,
            'actual': actual_vol,
            'predicted': pred_vol
        })

    df_garch = pd.DataFrame(garch_preds).set_index('date')
    return df_garch
```

EGARCH (Exponential GARCH) is a modification of GARCH that takes into account the asymmetric effects of volatility.

$$\log(\sigma_t^2) = \omega + \beta \cdot \log(\sigma_{t-1}^2) + \alpha \cdot |z_{t-1}| + \gamma \cdot z_{t-1}$$

Where:

- $z_{t-1} = \dfrac{\varepsilon_{t-1}}{\sigma_{t-1}}$ — standardized residual;
- $\omega$ — constant;
- $\beta$ — persistence coefficient;
- $\alpha$ — impact of shock magnitude;
- $\gamma$ — asymmetry parameter.

The code will be the same, except for the code of the model itself:

```python
model_garch = arch_model(hist_returns, vol='EGARCH', p=1, q=1, dist='normal')
```

GJR-GARCH is another model for asymmetric effects, named after its authors (Glosten, Jagannathan, and Runkle).

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \gamma \varepsilon_{t-1}^2 I_{t-1} + \beta \sigma_{t-1}^2$$

Where:

- $\sigma_t^2$ — conditional variance at time $t$;
- $\varepsilon_{t-1}$ — standardized residual at time $t-1$;
- $\omega$ — constant;
- $\beta$ — persistence coefficient;
- $\alpha$ — impact of shock magnitude;
- $\gamma$ — asymmetry parameter (leverage effect), which enhances the impact of negative shocks;
- $\varepsilon_{t-1}^2 I_{t-1}$ — indicator function, which is equal to 1 if $\varepsilon_{t-1}$ is negative, otherwise – 0.

The code will be the same, except for the code of the model itself:

```python
model_garch = arch_model(hist_returns, vol='GARCH', p=1, q=1, o=1, dist='normal')
```

o = 1 adds the GJR-GARCH effect.

The library handles all the mathematical implementation.

### 3.3 Metrics

The Mean Absolute Error (MAE) and Quasi-Likelihood (QLIKE) quality metrics were used.

Mean Absolute Error (MAE) is a regression model quality metric calculated as the arithmetic mean of the absolute values of the errors (the differences between the predicted and actual values).

Formula:

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Where:

- $y$ – true value
- $\hat{y}$ – predicted value

It will be taken from the python library scikit-learn:

```python
from sklearn.metrics import mean_absolute_error
```

Quasi-Likelihood (QLIKE) is a metric created specifically for evaluating volatility forecasts.

Formula:

$$QLIKE = \frac{1}{n} \sum_{i=1}^{n} \left(\log \hat{y}_i^2 + \frac{y_i^2}{\hat{y}_i^2}\right)$$

Where:

- $y$ – true value
- $\hat{y}$ – predicted value

There is no such function in scikit-learn, so it will be implemented manually:

```python
def qlike(y_true, y_pred):
    sigma2_true = y_true**2
    sigma2_pred = np.maximum(y_pred**2, 1e-10)
    return np.mean(np.log(sigma2_pred) + sigma2_true / sigma2_pred)
```

### 3.4 Test

After identifying the most accurate machine learning models and conditional heteroskedasticity, we will compare the results of their predictions on the Diebold-Mariano test, to identify the statistical superiority in the predictions of any of the models.

The Diebold-Mariano test is used to determine how much the 2 predictions differ. First, we calculate a series of errors. For MAE:

$$e_{1i} = |y_{1i} - f_{1i}|$$

$$e_{2i} = |y_{2i} - f_{2i}|$$

For QLIKE:

$$e_{1i} = \log f_{1i}^2 + \frac{y_{1i}^2}{f_{1i}^2}$$

$$e_{2i} = \log f_{2i}^2 + \frac{y_{2i}^2}{f_{2i}^2}$$

Where:

- $e$ – error series
- $y$ – true values series
- $f$ – predictions series

Next, we calculate the difference in loss between the two series.

$$d_i = e_{1i} - e_{2i}$$

If the difference in losses $d < 0$, then the forecast of the $e_1$ series is more accurate than the forecast of $e_2$.

Since we have a 1-step-ahead forecast, it is not necessary to consider autocorrelation.

The test formula without considering autocorrelation is:

$$DM = \frac{\sqrt{n} \cdot \dfrac{1}{n}\sum_{i=1}^{n} d_i}{\sqrt{\dfrac{1}{n-1}\sum_{i=1}^{n}(d_i - \bar{d})^2}}$$

We calculate the p-value:

$$p = 2 \cdot (1 - \Phi(|DM|))$$

Where $\Phi$ is the distribution function of the standard normal law:

$$\Phi(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-t^2/2} \, dt$$

The code:

```python
def dm_test(d):
    n = len(d)
    mean_d = np.mean(d)
    var_d = np.var(d)
    dm_stat = np.sqrt(n) * mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

err_dq = abs(the_best_dquant_aligned['actual'] - the_best_dquant_aligned['predicted'])
err_garch = abs(the_best_garch_aligned['actual'] - the_best_garch_aligned['predicted'])
d = err_dq - err_garch
dm_stat, p_value = dm_test(d)

dquant_sigma2_true = the_best_dquant_aligned_qlike['actual']**2
dquant_sigma2_pred = np.maximum(the_best_dquant_aligned_qlike['predicted']**2, 1e-10)
garch_sigma2_true = the_best_garch_aligned_qlike['actual']**2
garch_sigma2_pred = np.maximum(the_best_garch_aligned_qlike['predicted']**2, 1e-10)
err_dq = np.log(dquant_sigma2_pred) + dquant_sigma2_true/dquant_sigma2_pred
err_garch = np.log(garch_sigma2_pred) + garch_sigma2_true/garch_sigma2_pred
d = err_dq - err_garch
dm_stat_qlike, p_value_qlike = dm_test(d)
print(f"\n========== RESULTS {i} ==========")
print(f"Based on MAE results, for {i} the best GARCH model is {the_best_garch}, the best DQuant model is {the_best_dquant}")
print(f"MAE DM test: statistic = {dm_stat:.4f}, p-value = {p_value}")
if p_value < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")
print(f"Based on QLIKE results, for {i} the best GARCH model is {the_best_garch_qlike}, the best DQuant model is {the_best_dquant_qlike}")
print(f"QLIKE DM test: statistic = {dm_stat_qlike:.4f}, p-value = {p_value_qlike}")
if p_value_qlike < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")
```

### 3.5 The Code

All the source code is available on GitHub https://github.com/artrdon/DQuant_Research.

## 4 Results

### 4.1 Results when using the QLIKE loss function

**Table 1. Comparison of MAE and QLIKE for BTC-USD.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00910 | -6.47466 |
| LightGBM | 0.00888 | -6.20162 |
| GARCH | 0.01289 | -6.34743 |
| EGARCH | 0.01323 | -6.33964 |
| GJR-GARCH | 0.01277 | -6.35053 |

```text
========== RESULTS BTC-USD ==========
Based on MAE results, for BTC-USD the best GARCH model is GJR-GARCH, the best DQuant model is LightGBM
MAE DM test: statistics = -10.3584, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for BTC-USD the best GARCH model is GJR-GARCH, the best DQuant model is XGBoost
QLIKE DM test: statistics = -4.4082, p-value = 1.0423356946231976e-05
The difference is statistically significant.
```

**Table 2. Comparison of MAE and QLIKE for ETH-USD.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.01071 | -6.18312 |
| LightGBM | 0.00985 | -6.18634 |
| GARCH | 0.01531 | -6.01800 |
| EGARCH | 0.01463 | -6.05071 |
| GJR-GARCH | 0.01536 | -6.01689 |

```text
========== RESULTS ETH-USD ==========
Based on MAE results, for ETH-USD the best GARCH model is EGARCH, the best DQuant model is LightGBM
MAE DM test: statistics = -14.9522, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for ETH-USD the best GARCH model is EGARCH, the best DQuant model is LightGBM
QLIKE DM test: statistics = -5.8334, p-value = 5.42998490615787e-09
The difference is statistically significant.
```

**Table 3. Comparison of MAE and QLIKE for EURUSD=X.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00115 | -10.21415 |
| LightGBM | 0.00111 | -10.20654 |
| GARCH | 0.00140 | -10.21536 |
| EGARCH | 0.00144 | -10.20877 |
| GJR-GARCH | 0.00139 | -10.21929 |

```text
========== RESULTS EURUSD=X ==========
Based on MAE results, for EURUSD=X the best GARCH model is GJR-GARCH, the best DQuant model is LightGBM
MAE DM test: statistics = -6.7441, p-value = 1.5392798147217945e-11
The difference is statistically significant.
Based on QLIKE results, for EURUSD=X the best GARCH model is GJR-GARCH, the best DQuant model is XGBoost
QLIKE DM test: statistics = 0.2411, p-value = 0.8094884959577284
The difference is not statistically significant.
```

**Table 4. Comparison of MAE and QLIKE for BZ=F.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00537 | -7.20514 |
| LightGBM | 0.00440 | -7.20826 |
| GARCH | 0.00588 | -7.19007 |
| EGARCH | 0.00627 | -7.17082 |
| GJR-GARCH | 0.00601 | -7.18410 |

```text
========== RESULTS BZ=F ==========
Based on MAE results, for BZ=F the best GARCH model is GARCH, the best DQuant model is LightGBM
MAE DM test: statistics = -7.6751, p-value = 1.6431300764452317e-14
The difference is statistically significant.
Based on QLIKE results, for BZ=F the best GARCH model is GARCH, the best DQuant model is LightGBM
QLIKE DM test: statistics = -0.9704, p-value = 0.3318438760503746
The difference is not statistically significant.
```

**Table 5. Comparison of MAE and QLIKE for GC=F.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00292 | -9.15355 |
| LightGBM | 0.00278 | -8.88900 |
| GARCH | 0.00454 | -8.94346 |
| EGARCH | 0.00473 | -8.92604 |
| GJR-GARCH | 0.00454 | -8.94765 |

```text
========== RESULTS GC=F ==========
Based on MAE results, for GC=F the best GARCH model is GARCH, the best DQuant model is LightGBM
MAE DM test: statistics = -11.7233, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for GC=F the best GARCH model is GJR-GARCH, the best DQuant model is XGBoost
QLIKE DM test: statistics = -4.3960, p-value = 1.102764538596368e-05
The difference is statistically significant.
```

**Table 6. Comparison of MAE and QLIKE for SI=F.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00583 | -7.72549 |
| LightGBM | 0.00584 | -4.59948 |
| GARCH | 0.01110 | -7.67624 |
| EGARCH | 0.01156 | -7.63753 |
| GJR-GARCH | 0.01112 | -7.67493 |

```text
========== RESULTS SI=F ==========
Based on MAE results, for SI=F the best GARCH model is GARCH, the best DQuant model is XGBoost
MAE DM test: statistics = -15.6615, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for SI=F the best GARCH model is GARCH, the best DQuant model is XGBoost
QLIKE DM test: statistics = 0.3645, p-value = 0.7154501335422709
The difference is not statistically significant.
```

**Table 7. Comparison of MAE and QLIKE for SPY.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00170 | -9.30808 |
| LightGBM | 0.00160 | -9.27566 |
| GARCH | 0.00307 | -9.10276 |
| EGARCH | 0.00324 | -9.08008 |
| GJR-GARCH | 0.00295 | -9.13928 |

```text
========== RESULTS SPY ==========
Based on MAE results, for SPY the best GARCH model is GJR-GARCH, the best DQuant model is LightGBM
MAE DM test: statistics = -11.9430, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for SPY the best GARCH model is GJR-GARCH, the best DQuant model is XGBoost
QLIKE DM test: statistics = -4.6812, p-value = 2.8522799944141752e-06
The difference is statistically significant.
```

### 4.2 Results when using the MAE loss function

**Table 8. Comparison of MAE and QLIKE for BTC-USD.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00891 | -6.29434 |
| LightGBM | 0.00900 | -6.16667 |
| GARCH | 0.01289 | -6.34743 |
| EGARCH | 0.01323 | -6.33964 |
| GJR-GARCH | 0.01277 | -6.35053 |

```text
========== RESULTS BTC-USD ==========
Based on MAE results, for BTC-USD the best GARCH model is GJR-GARCH, the best DQuant model is XGBoost
MAE DM test: statistics = -10.8621, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for BTC-USD the best GARCH model is GJR-GARCH, the best DQuant model is XGBoost
QLIKE DM test: statistics = 1.1880, p-value = 0.23484011760514423
The difference is not statistically significant.
```

**Table 9. Comparison of MAE and QLIKE for ETH-USD.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.01083 | -6.20135 |
| LightGBM | 0.01076 | -6.20778 |
| GARCH | 0.01531 | -6.01800 |
| EGARCH | 0.01463 | -6.05071 |
| GJR-GARCH | 0.01536 | -6.01689 |

```text
========== RESULTS ETH-USD ==========
Based on MAE results, for ETH-USD the best GARCH model is EGARCH, the best DQuant model is LightGBM
MAE DM test: statistics = -15.1840, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for ETH-USD the best GARCH model is EGARCH, the best DQuant model is LightGBM
QLIKE DM test: statistics = -5.9318, p-value = 2.996629833162956e-09
The difference is statistically significant.
```

**Table 10. Comparison of MAE and QLIKE for EURUSD=X.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00111 | -10.21573 |
| LightGBM | 0.00111 | -10.21648 |
| GARCH | 0.00140 | -10.21536 |
| EGARCH | 0.00144 | -10.20877 |
| GJR-GARCH | 0.00139 | -10.21929 |

```text
========== RESULTS EURUSD=X ==========
Based on MAE results, for EURUSD=X the best GARCH model is GJR-GARCH, the best DQuant model is LightGBM
MAE DM test: statistics = -7.3202, p-value = 2.475797344914099e-13
The difference is statistically significant.
Based on QLIKE results, for EURUSD=X the best GARCH model is GJR-GARCH, the best DQuant model is LightGBM
QLIKE DM test: statistics = 0.1184, p-value = 0.9057742634310224
The difference is not statistically significant.
```

**Table 11. Comparison of MAE and QLIKE for BZ=F.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00422 | -7.23529 |
| LightGBM | 0.00436 | -7.23286 |
| GARCH | 0.00588 | -7.19007 |
| EGARCH | 0.00627 | -7.17082 |
| GJR-GARCH | 0.00601 | -7.18410 |

```text
========== RESULTS BZ=F ==========
Based on MAE results, for BZ=F the best GARCH model is GARCH, the best DQuant model is XGBoost
MAE DM test: statistics = -9.3536, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for BZ=F the best GARCH model is GARCH, the best DQuant model is XGBoost
QLIKE DM test: statistics = -2.6045, p-value = 0.009201599598327448
The difference is statistically significant.
```

**Table 12. Comparison of MAE and QLIKE for GC=F.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00273 | -8.92118 |
| LightGBM | 0.00274 | -8.93554 |
| GARCH | 0.00454 | -8.94346 |
| EGARCH | 0.00473 | -8.92604 |
| GJR-GARCH | 0.00454 | -8.94765 |

```text
========== RESULTS GC=F ==========
Based on MAE results, for GC=F the best GARCH model is GARCH, the best DQuant model is XGBoost
MAE DM test: statistics = -12.1707, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for GC=F the best GARCH model is GJR-GARCH, the best DQuant model is LightGBM
QLIKE DM test: statistics = 0.1163, p-value = 0.9074312953386849
The difference is not statistically significant.
```

**Table 13. Comparison of MAE and QLIKE for SI=F.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00577 | -5.51437 |
| LightGBM | 0.00583 | -5.22715 |
| GARCH | 0.01110 | -7.67624 |
| EGARCH | 0.01156 | -7.63753 |
| GJR-GARCH | 0.01112 | -7.67493 |

```text
========== RESULTS SI=F ==========
Based on MAE results, for SI=F the best GARCH model is GARCH, the best DQuant model is XGBoost
MAE DM test: statistics = -12.4694, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for SI=F the best GARCH model is GARCH, the best DQuant model is XGBoost
QLIKE DM test: statistics = 4.4522, p-value = 8.500008522371871e-06
The difference is statistically significant.
```

**Table 14. Comparison of MAE and QLIKE for SPY.**

| Model | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00159 | -9.26711 |
| LightGBM | 0.00155 | -9.29668 |
| GARCH | 0.00307 | -9.10277 |
| EGARCH | 0.00324 | -9.08008 |
| GJR-GARCH | 0.00295 | -9.13928 |

```text
========== RESULTS SPY ==========
Based on MAE results, for SPY the best GARCH model is GJR-GARCH, the best DQuant model is LightGBM
MAE DM test: statistics = -12.5817, p-value = 0.0
The difference is statistically significant.
Based on QLIKE results, for SPY the best GARCH model is GJR-GARCH, the best DQuant model is LightGBM
QLIKE DM test: statistics = -4.9101, p-value = 9.102401692384632e-07
The difference is statistically significant.
```

## 5 Conclusion

In this study, we compared conditional heteroskedasticity models from the arch library and machine learning models from the dquant library on 7 different financial instruments.

On the Diebold-Mariano test, when using MAE as the loss function, gradient boosting models showed a statistically significant advantage on all assets. This shows that machine learning models, particularly gradient boosting models, can capture nonlinear dependencies that GARCH models do not detect.

When using QLIKE as the loss function when training ML models, a statistical advantage was found for the MAE metric on all assets. For the QLIKE metric, a statistical advantage was found on 4 out of 7 assets. No statistical advantage was found on EURUSD=X, BZ=F, SI=F.

When using MAE as the loss function when training ML models, a statistical advantage was found for the MAE metric on all assets. For the QLIKE metric, a statistical advantage was found on 4 out of 7 assets. No statistical advantage was found for BTC-USD, EURUSD=X, GC=F. However, a statistical advantage was found for silver in favor of GARCH models.

The comparison was conducted on daily data, and the results may differ if intraday data is used.

Gradient boosting hyperparameters were not optimized, and the accuracy of machine learning models may be higher if optimal hyperparameters are used.

Only returns were used as a feature for machine learning models, and the accuracy of predictions may be higher if additional features are used.

There was no walk forward validation, but instead, the data was statically split into training and validation sets.

## 6 References

1. https://github.com/artrdon/dquant
2. https://mlgu.ru/1778/?utm_source=yandex&utm_medium=organic
3. https://real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/
4. https://elma365.com/ru/baza-znaniy/mse/
5. https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf
6. https://mlgu.ru/6807/?utm_source=yandex&utm_medium=organic
7. https://blog.quantinsti.com/garch-gjr-garch-volatility-forecasting-python/
8. https://ru.wikipedia.org/wiki/Волатильность
9. https://public.econ.duke.edu/~boller/Published_Papers/joe_86.pdf
