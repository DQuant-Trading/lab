<div align="center">
  <h3>Сравнение прогноза волатильности моделей машинного обучения и моделей условной гетероскедастичности</h3>
</div>
<div align="right">
  <h4>Денис Макаров</h4>
</div>
<div align="center">
  <h4>(04/06/2026)</h4>
  <h4>DQuant</h4>
  <h4>Ключевые слова: машинное обучение, волатильность, условная гетероскедастичность, прогноз волатильности</h4>
</div>

## 1 Аннотация

В работе сравнивается точность прогнозов волатильности моделей GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) и моделей машинного обучения XGBoost, LightGBM. Модели машинного обучения взяты из библиотеки dquant, а условной гетероскедастичности из библиотеки arch. Данные взяты с 01.01.2014 по 31.12.2024 из библиотеки yfinance. Тикеры: BTC-USD (биткоин/доллар), ETH-USD (эфир/доллар), EURUSD=X (евро/доллар), BZ=F (нефть), GC=F (золото), SI=F (серебро), SPY. Обучение проводилось на данных с 01.01.2014 до 01.01.2023, валидация с 01.01.2023 до 31.12.2024. Метрики качества прогнозов: QLIKE, MAE. Для проверки статистической значимости результатов используется тест Диболда-Мариано. Модели машинного обучения показали статистически значимое преимущество на всех активах по MAE и на 4 из 7 активах по QLIKE.

## 2 Введение

Волатильность — статистический финансовый показатель, характеризующий изменчивость цены на что-либо. Волатильность в финансах обычно определяется как стандартное отклонение или дисперсия доходности актива за определенный период времени.

Количественно определять степень изменчивости цены важно в трейдинге, волатильность используется везде: от риск-менеджмента до использования в торговых стратегиях, основанных на прогнозировании волатильности.

Революционный прорыв в моделировании волатильности был сделан Робертом Энглом в 1982 году, когда он предложил модель ARCH (Autoregressive Conditional Heteroskedasticity). В модели ARCH условная дисперсия в текущий момент времени представляется как линейная функция квадратов прошлых значений процесса.

Математически модель ARCH(q) можно представить следующим образом:

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2$$

где:

- $\sigma_t^2$ — условная дисперсия в момент времени $t$;
- $\omega$ — константа (обычно положительная);
- $\alpha_i$ — параметры модели, которые должны быть положительными для обеспечения положительности дисперсии;
- $\varepsilon_{t-i}$ — инновации процесса (обычно предполагается, что они независимы и одинаково распределены с нулевым средним).

За разработку модели ARCH Роберт Энгл был удостоен Нобелевской премии по экономике в 2003 году, что подчеркивает значимость этого вклада в финансовую эконометрику.

В 1986 году Тим Болерслев предложил обобщение модели ARCH, которое получило название GARCH (Generalized Autoregressive Conditional Heteroskedasticity). Ключевая идея заключалась в том, чтобы включить прошлые условные дисперсии в уравнение для текущей условной дисперсии:

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2$$

где $\beta_j$ — параметры, отражающие влияние прошлых условных дисперсий.

Эта формулировка оказалась более экономной в параметрах и часто модель GARCH(1,1) была достаточна для адекватного описания волатильности, тогда как эквивалентная модель ARCH потребовала бы бесконечного числа параметров.

Модели машинного обучения позволяют обрабатывать огромное количество данных, они используются в том числе и в трейдинге, в частности и в прогнозировании волатильности. Для этого в большинстве случаев используют бустинговые модели машинного обучения, они не требуют слишком много данных как нейронные сети, но при этом их прогнозы не отстают от прогнозов нейросетевых моделей.

Цель данной работы – сравнить точность прогнозов волатильности на 1 шаг вперед на 7 различных активах, используя модели семейства GARCH и модели градиентного бустинга. GARCH модели будут взяты из библиотеки arch, модели градиентного бустинга – из авторской библиотеки dquant. Точность прогнозов будут сравниваться на метриках MAE и QLIKE. Статистическая значимость самых точных моделей будет проверятся на тесте Диболда-Мариано.

## 3 Методология

### 3.1 Данные

Данные взяты из python библиотеки yfinance. Для обучения был взят промежуток в 9 лет OHLC данных дневного тайм фрейма. Для валидации – 2 года. Используются следующие активы: BTC-USD (биткоин/доллар), ETH-USD (эфир/доллар), EURUSD=X (евро/доллар), BZ=F (нефть), GC=F (золото), SI=F (серебро), SPY.

Важно использовать разные рынки с разными движениями цен. Например, если уже есть пара евро/доллар, то пару фунт/доллар использовать не обязательно. Из-за сильной корреляции этих двух инструментов, результаты будут похожими.

После получения данных нужно их разделить на тренировочные и валидационные. На тренировочных данных будут обучаться наши модели. Валидационные данные будут использоваться только после обучения. Код на python предоставляется:

```python
START_DATE = '2014-01-01'
SPLIT_DATE = '2023-01-01'
END_DATE = '2024-12-31'

def get_data(ticker, start_date, end_date):
    print(f"Загрузка {ticker} с {start_date} по {end_date}...")
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

### 3.2 Используемые модели, их обучение и валидация

#### 3.2.1 Модели машинного обучения

Будут использованы модели машинного обучения XGBoost и LightGBM из python библиотеки DQuant. Эта библиотека представляет удобный интерфейс для обучения ML моделей для прогноза волатильности, она была создана специально для этого.

Мы будем использовать доходности в качестве входных данных и волатильность Паркинсона в качестве таргетов. Формула волатильности Паркинсона:

$$\sigma_p = \sqrt{\frac{1}{4n \ln 2} \sum_{i=1}^{n} \left(\ln \frac{H_i}{L_i}\right)^2}$$

В библиотеке реализован удобный интерфейс для создания признаков и таргетов, но для наглядности мы реализуем эти функции вручную, библиотека позволяет встраивать свои функции для создания признаков и таргетов.

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

Встраиваем эти функции в код обучения модели:

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

Обучение будет проходить на функции потерь QLIKE, которая была создана специально для обучения прогнозировать волатильность и на MAE.

Обучение проводилось с гиперпараметрами по умолчанию. Для XGBoost это:

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

Для LightGBM:

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

Оптимальное количество деревьев подбирается в процессе обучения.

Код для получения прогнозов модели:

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
            print(f"Ошибка прогноза DQuant на {test_date.date()}: {e}")
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

Все то же самое нужно сделать и с VolClustLightGBM.

#### 3.2.2 Модели условной гетероскедастичности

Будут использованы модели условной гетероскедастичности: GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) из python библиотеки arch.

GARCH (Generalized Autoregressive Conditional Heteroskedasticity) – модель условной гетероскедастичности.

Формула выглядит следующим образом:

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Где:

- $\sigma_t^2$ — условная дисперсия в момент $t$;
- $\varepsilon_{t-1}$ — стандартизированный остаток в момент $t-1$;
- $\omega$ — константа;
- $\beta$ — коэффициент персистентности;
- $\alpha$ — влияние величины шока.

Реализация в python:

```python
def garch_forecasting():
    garch_preds = []

    returns_full = df['returns'].dropna() * 100   # масштабирование для GARCH
    for test_date in df_test.index:
        hist_returns = returns_full[:test_date].iloc[:-1]
        if len(hist_returns) < 500:
            continue

        try:
            model_garch = arch_model(hist_returns, vol='GARCH', p=1, q=1, dist='normal')
            fitted = model_garch.fit(disp='off')
            forec = fitted.forecast(horizon=HORIZON, reindex=False)
            cond_var = forec.variance.values[-1, 0]
            pred_vol = np.sqrt(cond_var) / 100.0   # обратное масштабирование
        except Exception as e:
            print(f"Ошибка GARCH на {test_date.date()}: {e}")
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

EGARCH (Exponential GARCH) – модификация GARCH, которая учитывает асимметричные эффекты волатильности.

$$\log(\sigma_t^2) = \omega + \beta \cdot \log(\sigma_{t-1}^2) + \alpha \cdot |z_{t-1}| + \gamma \cdot z_{t-1}$$

Где:

- $z_{t-1} = \dfrac{\varepsilon_{t-1}}{\sigma_{t-1}}$ — стандартизированный остаток;
- $\omega$ — константа;
- $\beta$ — коэффициент персистентности;
- $\alpha$ — влияние величины шока;
- $\gamma$ — параметр асимметрии.

Код будет тот же самый, кроме кода самой модели:

```python
model_garch = arch_model(hist_returns, vol='EGARCH', p=1, q=1, dist='normal')
```

GJR-GARCH – еще одна модель для асимметричных эффектов, названная в честь ее авторов (Glosten, Jagannathan, и Runkle).

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \gamma \varepsilon_{t-1}^2 I_{t-1} + \beta \sigma_{t-1}^2$$

Где:

- $\sigma_t^2$ — условная дисперсия в момент $t$;
- $\varepsilon_{t-1}$ — стандартизированный остаток в момент $t-1$;
- $\omega$ — константа;
- $\beta$ — коэффициент персистентности;
- $\alpha$ — влияние величины шока;
- $\gamma$ — параметр асимметрии (леверидж-эффект), который усиливает влияние отрицательных шоков;
- $\varepsilon_{t-1}^2 I_{t-1}$ — индикаторная функция, которая равна 1, если $\varepsilon_{t-1}$ отрицательный, иначе – 0.

Код будет тот же самый, кроме кода самой модели:

```python
model_garch = arch_model(hist_returns, vol='GARCH', p=1, q=1, o=1, dist='normal')
```

o = 1 добавляет эффект GJR-GARCH.

Библиотека берет реализацию всей математики на себя.

### 3.3 Метрики

Использованы метрики качества Mean Absolute Error (MAE) и Quasi-Likelihood (QLIKE).

Mean Absolute Error (MAE) – это метрика оценки качества моделей регрессии, вычисляемая как среднее арифметическое абсолютных значений ошибок (разностей между предсказанными и реальными значениями).

Формула:

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Где:

- $y$ – истинное значение
- $\hat{y}$ – прогнозируемое значение

Она будет взята из python библиотеки scikit-learn:

```python
from sklearn.metrics import mean_absolute_error
```

Quasi-Likelihood (QLIKE) – метрика, созданная специально для оценки прогнозов волатильности.

Формула:

$$QLIKE = \frac{1}{n} \sum_{i=1}^{n} \left(\log \hat{y}_i^2 + \frac{y_i^2}{\hat{y}_i^2}\right)$$

Где:

- $y$ – истинное значение
- $\hat{y}$ – прогнозируемое значение

В scikit-learn нет такой функции, поэтому она будет реализована вручную:

```python
def qlike(y_true, y_pred):
    sigma2_true = y_true**2
    sigma2_pred = np.maximum(y_pred**2, 1e-10)
    return np.mean(np.log(sigma2_pred) + sigma2_true / sigma2_pred)
```

### 3.4 Тест

После выявления самых точных моделей машинного обучения и условной гетероскедастичности, сравним результаты их прогнозов на тесте Диболда-Мариано, для выявления статистического превосходства в прогнозах какой-либо из моделей.

Тест Диболда-Мариано используется для определения на сколько сильно отличаются 2 прогноза. Для начала мы вычисляем ряд ошибок. Для MAE:

$$e_{1i} = |y_{1i} - f_{1i}|$$

$$e_{2i} = |y_{2i} - f_{2i}|$$

Для QLIKE:

$$e_{1i} = \log f_{1i}^2 + \frac{y_{1i}^2}{f_{1i}^2}$$

$$e_{2i} = \log f_{2i}^2 + \frac{y_{2i}^2}{f_{2i}^2}$$

Где:

- $e$ – ряд ошибок
- $y$ – ряд истинных значений
- $f$ – ряд прогнозов

Далее мы вычисляем разницу в потерях между двумя рядами.

$$d_i = e_{1i} - e_{2i}$$

Если разница в потерях $d < 0$, то прогноз ряда $e_1$ точнее, чем прогноз $e_2$.

Так как у нас прогноз на 1 шаг вперед, то учитывать автокорреляцию не обязательно.

Формула теста без учета автокорреляции:

$$DM = \frac{\sqrt{n} \cdot \dfrac{1}{n}\sum_{i=1}^{n} d_i}{\sqrt{\dfrac{1}{n-1}\sum_{i=1}^{n}(d_i - \bar{d})^2}}$$

Вычисляем p-value:

$$p = 2 \cdot (1 - \Phi(|DM|))$$

Где $\Phi$ – функция распределения стандартного нормального закона:

$$\Phi(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-t^2/2} \, dt$$

Код:

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
print(f"\n========== РЕЗУЛЬТАТЫ {i} ==========")
print(f"По результатам MAE, для {i} лучшая GARCH модель – {the_best_garch}, лучшая DQuant модель – {the_best_dquant}")
print(f"MAE DM-тест: статистика = {dm_stat:.4f}, p-value = {p_value}")
if p_value < 0.05:
    print("Различие статистически значимо.")
else:
    print("Различие не является статистически значимым.")
print(f"По результатам QLIKE, для {i} лучшая GARCH модель – {the_best_garch_qlike}, лучшая DQuant модель – {the_best_dquant_qlike}")
print(f"QLIKE DM-тест: статистика = {dm_stat_qlike:.4f}, p-value = {p_value_qlike}")
if p_value_qlike < 0.05:
    print("Различие статистически значимо.")
else:
    print("Различие не является статистически значимым.")
```

### 3.5 Код

Весь исходный код есть на GitHub https://github.com/artrdon/DQuant_Research.

## 4 Результаты

### 4.1 Результаты при использовании функции потерь QLIKE

**Таблица 1. Сравнение MAE и QLIKE для BTC-USD.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00910 | -6.47466 |
| LightGBM | 0.00888 | -6.20162 |
| GARCH | 0.01289 | -6.34743 |
| EGARCH | 0.01323 | -6.33964 |
| GJR-GARCH | 0.01277 | -6.35053 |

```text
========== РЕЗУЛЬТАТЫ BTC-USD ==========
По результатам MAE, для BTC-USD лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – LightGBM
MAE DM-тест: статистика = -10.3584, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для BTC-USD лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – XGBoost
QLIKE DM-тест: статистика = -4.4082, p-value = 1.0423356946231976e-05
Различие статистически значимо.
```

**Таблица 2. Сравнение MAE и QLIKE для ETH-USD.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.01071 | -6.18312 |
| LightGBM | 0.00985 | -6.18634 |
| GARCH | 0.01531 | -6.01800 |
| EGARCH | 0.01463 | -6.05071 |
| GJR-GARCH | 0.01536 | -6.01689 |

```text
========== РЕЗУЛЬТАТЫ ETH-USD ==========
По результатам MAE, для ETH-USD лучшая GARCH модель – EGARCH, лучшая DQuant модель – LightGBM
MAE DM-тест: статистика = -14.9522, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для ETH-USD лучшая GARCH модель – EGARCH, лучшая DQuant модель – LightGBM
QLIKE DM-тест: статистика = -5.8334, p-value = 5.42998490615787e-09
Различие статистически значимо.
```

**Таблица 3. Сравнение MAE и QLIKE для EURUSD=X.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00115 | -10.21415 |
| LightGBM | 0.00111 | -10.20654 |
| GARCH | 0.00140 | -10.21536 |
| EGARCH | 0.00144 | -10.20877 |
| GJR-GARCH | 0.00139 | -10.21929 |

```text
========== РЕЗУЛЬТАТЫ EURUSD=X ==========
По результатам MAE, для EURUSD=X лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – LightGBM
MAE DM-тест: статистика = -6.7441, p-value = 1.5392798147217945e-11
Различие статистически значимо.
По результатам QLIKE, для EURUSD=X лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – XGBoost
QLIKE DM-тест: статистика = 0.2411, p-value = 0.8094884959577284
Различие не является статистически значимым.
```

**Таблица 4. Сравнение MAE и QLIKE для BZ=F.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00537 | -7.20514 |
| LightGBM | 0.00440 | -7.20826 |
| GARCH | 0.00588 | -7.19007 |
| EGARCH | 0.00627 | -7.17082 |
| GJR-GARCH | 0.00601 | -7.18410 |

```text
========== РЕЗУЛЬТАТЫ BZ=F ==========
По результатам MAE, для BZ=F лучшая GARCH модель – GARCH, лучшая DQuant модель – LightGBM
MAE DM-тест: статистика = -7.6751, p-value = 1.6431300764452317e-14
Различие статистически значимо.
По результатам QLIKE, для BZ=F лучшая GARCH модель – GARCH, лучшая DQuant модель – LightGBM
QLIKE DM-тест: статистика = -0.9704, p-value = 0.3318438760503746
Различие не является статистически значимым.
```

**Таблица 5. Сравнение MAE и QLIKE для GC=F.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00292 | -9.15355 |
| LightGBM | 0.00278 | -8.88900 |
| GARCH | 0.00454 | -8.94346 |
| EGARCH | 0.00473 | -8.92604 |
| GJR-GARCH | 0.00454 | -8.94765 |

```text
========== РЕЗУЛЬТАТЫ GC=F ==========
По результатам MAE, для GC=F лучшая GARCH модель – GARCH, лучшая DQuant модель – LightGBM
MAE DM-тест: статистика = -11.7233, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для GC=F лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – XGBoost
QLIKE DM-тест: статистика = -4.3960, p-value = 1.102764538596368e-05
Различие статистически значимо.
```

**Таблица 6. Сравнение MAE и QLIKE для SI=F.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00583 | -7.72549 |
| LightGBM | 0.00584 | -4.59948 |
| GARCH | 0.01110 | -7.67624 |
| EGARCH | 0.01156 | -7.63753 |
| GJR-GARCH | 0.01112 | -7.67493 |

```text
========== РЕЗУЛЬТАТЫ SI=F ==========
По результатам MAE, для SI=F лучшая GARCH модель – GARCH, лучшая DQuant модель – XGBoost
MAE DM-тест: статистика = -15.6615, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для SI=F лучшая GARCH модель – GARCH, лучшая DQuant модель – XGBoost
QLIKE DM-тест: статистика = 0.3645, p-value = 0.7154501335422709
Различие не является статистически значимым.
```

**Таблица 7. Сравнение MAE и QLIKE для SPY.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00170 | -9.30808 |
| LightGBM | 0.00160 | -9.27566 |
| GARCH | 0.00307 | -9.10276 |
| EGARCH | 0.00324 | -9.08008 |
| GJR-GARCH | 0.00295 | -9.13928 |

```text
========== РЕЗУЛЬТАТЫ SPY ==========
По результатам MAE, для SPY лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – LightGBM
MAE DM-тест: статистика = -11.9430, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для SPY лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – XGBoost
QLIKE DM-тест: статистика = -4.6812, p-value = 2.8522799944141752e-06
Различие статистически значимо.
```

### 4.2 Результаты при использовании функции потерь MAE

**Таблица 8. Сравнение MAE и QLIKE для BTC-USD.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00891 | -6.29434 |
| LightGBM | 0.00900 | -6.16667 |
| GARCH | 0.01289 | -6.34743 |
| EGARCH | 0.01323 | -6.33964 |
| GJR-GARCH | 0.01277 | -6.35053 |

```text
========== РЕЗУЛЬТАТЫ BTC-USD ==========
По результатам MAE, для BTC-USD лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – XGBoost
MAE DM-тест: статистика = -10.8621, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для BTC-USD лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – XGBoost
QLIKE DM-тест: статистика = 1.1880, p-value = 0.23484011760514423
Различие не является статистически значимым.
```

**Таблица 9. Сравнение MAE и QLIKE для ETH-USD.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.01083 | -6.20135 |
| LightGBM | 0.01076 | -6.20778 |
| GARCH | 0.01531 | -6.01800 |
| EGARCH | 0.01463 | -6.05071 |
| GJR-GARCH | 0.01536 | -6.01689 |

```text
========== РЕЗУЛЬТАТЫ ETH-USD ==========
По результатам MAE, для ETH-USD лучшая GARCH модель – EGARCH, лучшая DQuant модель – LightGBM
MAE DM-тест: статистика = -15.1840, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для ETH-USD лучшая GARCH модель – EGARCH, лучшая DQuant модель – LightGBM
QLIKE DM-тест: статистика = -5.9318, p-value = 2.996629833162956e-09
Различие статистически значимо.
```

**Таблица 10. Сравнение MAE и QLIKE для EURUSD=X.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00111 | -10.21573 |
| LightGBM | 0.00111 | -10.21648 |
| GARCH | 0.00140 | -10.21536 |
| EGARCH | 0.00144 | -10.20877 |
| GJR-GARCH | 0.00139 | -10.21929 |

```text
========== РЕЗУЛЬТАТЫ EURUSD=X ==========
По результатам MAE, для EURUSD=X лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – LightGBM
MAE DM-тест: статистика = -7.3202, p-value = 2.475797344914099e-13
Различие статистически значимо.
По результатам QLIKE, для EURUSD=X лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – LightGBM
QLIKE DM-тест: статистика = 0.1184, p-value = 0.9057742634310224
Различие не является статистически значимым.
```

**Таблица 11. Сравнение MAE и QLIKE для BZ=F.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00422 | -7.23529 |
| LightGBM | 0.00436 | -7.23286 |
| GARCH | 0.00588 | -7.19007 |
| EGARCH | 0.00627 | -7.17082 |
| GJR-GARCH | 0.00601 | -7.18410 |

```text
========== РЕЗУЛЬТАТЫ BZ=F ==========
По результатам MAE, для BZ=F лучшая GARCH модель – GARCH, лучшая DQuant модель – XGBoost
MAE DM-тест: статистика = -9.3536, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для BZ=F лучшая GARCH модель – GARCH, лучшая DQuant модель – XGBoost
QLIKE DM-тест: статистика = -2.6045, p-value = 0.009201599598327448
Различие статистически значимо.
```

**Таблица 12. Сравнение MAE и QLIKE для GC=F.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00273 | -8.92118 |
| LightGBM | 0.00274 | -8.93554 |
| GARCH | 0.00454 | -8.94346 |
| EGARCH | 0.00473 | -8.92604 |
| GJR-GARCH | 0.00454 | -8.94765 |

```text
========== РЕЗУЛЬТАТЫ GC=F ==========
По результатам MAE, для GC=F лучшая GARCH модель – GARCH, лучшая DQuant модель – XGBoost
MAE DM-тест: статистика = -12.1707, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для GC=F лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – LightGBM
QLIKE DM-тест: статистика = 0.1163, p-value = 0.9074312953386849
Различие не является статистически значимым.
```

**Таблица 13. Сравнение MAE и QLIKE для SI=F.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00577 | -5.51437 |
| LightGBM | 0.00583 | -5.22715 |
| GARCH | 0.01110 | -7.67624 |
| EGARCH | 0.01156 | -7.63753 |
| GJR-GARCH | 0.01112 | -7.67493 |

```text
========== РЕЗУЛЬТАТЫ SI=F ==========
По результатам MAE, для SI=F лучшая GARCH модель – GARCH, лучшая DQuant модель – XGBoost
MAE DM-тест: статистика = -12.4694, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для SI=F лучшая GARCH модель – GARCH, лучшая DQuant модель – XGBoost
QLIKE DM-тест: статистика = 4.4522, p-value = 8.500008522371871e-06
Различие статистически значимо.
```

**Таблица 14. Сравнение MAE и QLIKE для SPY.**

| Модель | MAE | QLIKE |
| --- | --- | --- |
| XGBoost | 0.00159 | -9.26711 |
| LightGBM | 0.00155 | -9.29668 |
| GARCH | 0.00307 | -9.10277 |
| EGARCH | 0.00324 | -9.08008 |
| GJR-GARCH | 0.00295 | -9.13928 |

```text
========== РЕЗУЛЬТАТЫ SPY ==========
По результатам MAE, для SPY лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – LightGBM
MAE DM-тест: статистика = -12.5817, p-value = 0.0
Различие статистически значимо.
По результатам QLIKE, для SPY лучшая GARCH модель – GJR-GARCH, лучшая DQuant модель – LightGBM
QLIKE DM-тест: статистика = -4.9101, p-value = 9.102401692384632e-07
Различие статистически значимо.
```

## 5 Заключение

В данном исследовании было проведено сравнение моделей условной гетероскедастичности, взятых из библиотеки arch и моделей машинного обучения из библиотеки dquant на 7 различных финансовых инструментах.

На тесте Диболда-Мариано, при использовании MAE в качестве функции потерь, модели градиентного бустинга показали статистически значимое преимущество на всех активах. Это показывает, что модели машинного обучения, в частности градиентного бустинга, могут улавливать нелинейные зависимости, которые не замечают модели семейства GARCH.

При использовании QLIKE в качестве функции потерь при обучении ML моделей, статистическое преимущество по метрике MAE было выявлено на всех активах. По метрике QLIKE – на 4 из 7 активах. Не было выявлено статистического преимущества на EURUSD=X, BZ=F, SI=F.

При использовании MAE в качестве функции потерь при обучении ML моделей, статистическое преимущество по метрике MAE было выявлено на всех активах. По метрике QLIKE – на 4 из 7 активах. Не было выявлено статистического преимущества на BTC-USD, EURUSD=X, GC=F. Причем, на серебре было выявлено статистическое преимущество в пользу моделей GARCH.

Сравнение проводилось на дневных данных, результаты могут отличаться, если брать внутридневные данные.

Не проводилась оптимизация гиперпараметров градиентного бустинга, точность прогнозов моделей машинного обучения может быть выше, если использовать оптимальные гиперпараметры.

В качестве признаков для моделей машинного обучения была только доходность, точность прогнозом может быть выше, если использовать дополнительные признаки.

Не проводилась walk forward валидация, вместо этого было статическое разделение данных на тренировочные и валидационные выборки.

## 6 Источники

1. https://github.com/artrdon/dquant
2. https://mlgu.ru/1778/?utm_source=yandex&utm_medium=organic
3. https://real-statistics.com/time-series-analysis/forecasting-accuracy/diebold-mariano-test/
4. https://elma365.com/ru/baza-znaniy/mse/
5. https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf
6. https://mlgu.ru/6807/?utm_source=yandex&utm_medium=organic
7. https://blog.quantinsti.com/garch-gjr-garch-volatility-forecasting-python/
8. https://ru.wikipedia.org/wiki/Волатильность
9. https://public.econ.duke.edu/~boller/Published_Papers/joe_86.pdf
