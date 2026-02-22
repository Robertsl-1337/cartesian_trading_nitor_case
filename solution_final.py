### Cartesian Trading 
### Sebastian Poshteh & Robert Sølyst Lildholdt 

"""
NITOR ENERGY — Day-ahead Electricity Price Forecasting

20-model seed-diversity ensemble with per-market SLSQP weight optimisation.

Models: 
  pm_lgb      : Per-market LightGBM blend (deep + regularised), seed=42
  pm_xgb      : Per-market XGBoost depth=7, seed=42
  g_xgb_d7_*  : Global XGBoost depth=7, 7 seeds (42,123,456,789,999,314,2024)
  g_xgb_d8_*  : Global XGBoost depth=8, 5 seeds (42,123,456,789,999)
  g_lgb_d127  : Global LightGBM num_leaves=127, seed=123
  g_lgb_d255  : Global LightGBM num_leaves=255, seed=123
  g_dart_*    : Global LightGBM DART, 4 seeds (42,123,456,789)

Ensemble: SLSQP minimises per-market RMSE on holdout-2024, 100 random starts.
Retrain:  All models retrained on the full training set before final prediction.

DATA LEAKAGE AUDIT
------------------
  Target encoding   : Excludes Sep-Nov 2023 and Sep-Nov 2024 from TE source.
                      Uses only data from the 2 years prior to the max training
                      date that do NOT overlap with either holdout window.
  1-year lags       : Sources are historical targets (364-367 days back).
                      For Sep-Nov 2025 test period, lags come from Sep-Nov 2024
                      which is present in the training CSV.  No future data.
  Day-level features: Aggregated within each (date, market) group independently
                      for train and test.  No cross-contamination.
  Cross-market feats: Aggregated per delivery_start timestamp, computed
                      separately on train and test.  No leakage.
  Intraday features : 3-hour rolling window within same (date, market).
                      Applied independently per dataframe.
  Regime clustering : StandardScaler + KMeans fitted on training split only
                      (tr_idx), applied via .transform() to holdout and test.
  Winsorisation     : Percentile bounds computed from training split only.
  Sample weights    : Exponential decay from delivery_start date — no fitting.
  SLSQP weights     : Optimised on holdout-2024 OOF predictions only.
  Retrain eval set  : Last 8% of chronologically sorted all-market data used
                      for early stopping only — does not affect learned weights.

REPRODUCIBILITY
---------------
  All random seeds are fixed (see SEEDS section below).
  np.random.seed(RANDOM_SEED) is set globally.
  Running this script twice on the same machine produces identical output.
"""

import os, warnings, gc, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# SEEDS — single source of truth for all randomness

RANDOM_SEED = 42          
np.random.seed(RANDOM_SEED)

D7_SEEDS   = [42, 123, 456, 789, 999, 314, 2024]   # 7 global XGB depth-7 models
D8_SEEDS   = [42, 123, 456, 789, 999]               # 5 global XGB depth-8 models
DART_SEEDS = [42, 123, 456, 789]                    # 4 global LGB DART models
LGB_D127_SEED = 123                                 # global LGB num_leaves=127
LGB_D255_SEED = 123                                 # global LGB num_leaves=255
PM_LGB_SEED   = 42                                  # per-market LGB
PM_XGB_SEED   = 42                                  # per-market XGB
KMEANS_SEED   = 42                                  # regime clustering
SLSQP_SEED    = 42                                  # SLSQP random starts

N_SLSQP_STARTS = 100       # random starts per market in SLSQP

t0 = time.time()
def rmse(a, b): return float(np.sqrt(mean_squared_error(np.array(a), np.array(b))))

# =============================================================================
# 1. LOAD DATA
# =============================================================================
train  = pd.read_csv('train.csv')
test   = pd.read_csv('test_for_participants.csv')
sample = pd.read_csv('sample_submission.csv')

for df in [train, test]:
    df['delivery_start'] = pd.to_datetime(df['delivery_start'])
    df['hour']           = df['delivery_start'].dt.hour
    df['dow']            = df['delivery_start'].dt.dayofweek
    df['month']          = df['delivery_start'].dt.month
    df['year']           = df['delivery_start'].dt.year
    df['doy']            = df['delivery_start'].dt.dayofyear
    df['hour_of_week']   = df['dow'] * 24 + df['hour']
    df['week']           = df['delivery_start'].dt.isocalendar().week.astype(int)

MARKETS = sorted(train['market'].unique())
print(f"Markets: {MARKETS}")
print(f"Train rows: {len(train):,}  Test rows: {len(test):,}")
print(f"Train period: {train.delivery_start.min().date()} -> {train.delivery_start.max().date()}")
print(f"Test period:  {test.delivery_start.min().date()} -> {test.delivery_start.max().date()}")

# =============================================================================
# 2. OUTLIER REMOVAL
# =============================================================================
mask_out = train['target'].abs() > 500
for mkt in MARKETS:
    m = train['market'] == mkt
    mu, sig = train.loc[m, 'target'].mean(), train.loc[m, 'target'].std()
    mask_out |= (m & ((train['target'] - mu).abs() > 4.5 * sig))
n_removed = mask_out.sum()
train = train[~mask_out].reset_index(drop=True)
print(f"\nRemoved {n_removed} outliers. Target range: [{train.target.min():.1f}, {train.target.max():.1f}]")

# =============================================================================
# 3. DAY-LEVEL FEATURES
#    Aggregated within (date, market) — no cross-date or cross-set leakage.
# =============================================================================
def add_day_features(df):
    d = df.copy()
    d['_date'] = d['delivery_start'].dt.date
    agg = d.groupby(['_date', 'market']).agg(
        day_solar_sum = ('solar_forecast',               'sum'),
        day_wind_sum  = ('wind_forecast',                'sum'),
        day_load_sum  = ('load_forecast',                'sum'),
        day_temp_avg  = ('air_temperature_2m',           'mean'),
        day_temp_min  = ('air_temperature_2m',           'min'),
        day_wind_max  = ('wind_speed_80m',               'max'),
        day_cloud_avg = ('cloud_cover_total',            'mean'),
        day_rain_sum  = ('precipitation_amount',         'sum'),
        day_solar_max = ('global_horizontal_irradiance', 'max'),
    ).reset_index()
    agg['day_balance']   = agg['day_load_sum'] - agg['day_solar_sum'] - agg['day_wind_sum']
    agg['day_renew_pct'] = (agg['day_solar_sum'] + agg['day_wind_sum']) / (agg['day_load_sum'] + 1)
    agg['day_hdd']       = np.maximum(0, 18 - agg['day_temp_avg'])
    agg['day_cdd']       = np.maximum(0, agg['day_temp_avg'] - 22)
    agg['day_is_rainy']  = (agg['day_rain_sum'] > 5).astype(int)
    agg['day_is_windy']  = (agg['day_wind_max'] > 12).astype(int)
    agg['day_is_sunny']  = (agg['day_solar_max'] > 400).astype(int)
    return d.merge(agg, on=['_date', 'market'], how='left').drop('_date', axis=1)

train = add_day_features(train)
test  = add_day_features(test)

# =============================================================================
# 4. TARGET ENCODING
#    Source: training rows where month NOT in (Sep, Oct, Nov) for years 2023/2024,
#    and within the 2-year window before the training cutoff.
#    → Sep-Nov 2023 and Sep-Nov 2024 holdout rows are NEVER in the TE source.
# =============================================================================
mask_te_src = ~(train['month'].isin([9, 10, 11]) & train['year'].isin([2023, 2024]))
te_cutoff   = train['delivery_start'].max() - pd.Timedelta(days=730)
recent_te   = train[mask_te_src & (train['delivery_start'] >= te_cutoff)]

G_MEAN = recent_te['target'].mean()
G_STD  = recent_te['target'].std()

te_hdow  = recent_te.groupby(['market', 'hour', 'dow'])['target'].agg(['mean', 'std', 'median'])
te_hdow.columns = ['te_hdow_mean', 'te_hdow_std', 'te_hdow_med']
te_hmo   = recent_te.groupby(['market', 'hour', 'month'])['target'].mean().rename('te_hmo_mean')
te_h     = recent_te.groupby(['market', 'hour'])['target'].mean().rename('te_h_mean')
te_mkt   = recent_te.groupby('market')['target'].mean().rename('te_mkt_mean')
te_mktmo = recent_te.groupby(['market', 'month'])['target'].mean().rename('te_mktmo_mean')
te_q90   = recent_te.groupby(['market', 'hour', 'dow'])['target'].quantile(0.90).rename('te_q90')
te_q10   = recent_te.groupby(['market', 'hour', 'dow'])['target'].quantile(0.10).rename('te_q10')
te_how   = recent_te.groupby(['market', 'hour_of_week'])['target'].mean().rename('te_how_mean')
te_wk    = recent_te.groupby(['market', 'week'])['target'].mean().rename('te_wk_mean')

def add_te(df):
    df = df.copy()
    df = df.join(te_hdow,  on=['market', 'hour', 'dow'])
    df = df.join(te_hmo,   on=['market', 'hour', 'month'])
    df = df.join(te_h,     on=['market', 'hour'])
    df = df.join(te_mkt,   on='market')
    df = df.join(te_mktmo, on=['market', 'month'])
    df = df.join(te_q90,   on=['market', 'hour', 'dow'])
    df = df.join(te_q10,   on=['market', 'hour', 'dow'])
    df = df.join(te_how,   on=['market', 'hour_of_week'])
    df = df.join(te_wk,    on=['market', 'week'])
    df['te_hdow_mean']  = df['te_hdow_mean'].fillna(df['te_h_mean']).fillna(G_MEAN)
    df['te_hdow_std']   = df['te_hdow_std'].fillna(G_STD)
    df['te_hdow_med']   = df['te_hdow_med'].fillna(G_MEAN)
    df['te_hmo_mean']   = df['te_hmo_mean'].fillna(df['te_h_mean']).fillna(G_MEAN)
    df['te_h_mean']     = df['te_h_mean'].fillna(G_MEAN)
    df['te_mkt_mean']   = df['te_mkt_mean'].fillna(G_MEAN)
    df['te_mktmo_mean'] = df['te_mktmo_mean'].fillna(df['te_mkt_mean']).fillna(G_MEAN)
    df['te_q90']        = df['te_q90'].fillna(G_MEAN + G_STD)
    df['te_q10']        = df['te_q10'].fillna(G_MEAN - G_STD)
    df['te_how_mean']   = df['te_how_mean'].fillna(df['te_h_mean']).fillna(G_MEAN)
    df['te_wk_mean']    = df['te_wk_mean'].fillna(df['te_mkt_mean']).fillna(G_MEAN)
    df['te_hour_prem']  = df['te_hdow_mean'] - df['te_mkt_mean']
    df['te_seas_ratio'] = df['te_hmo_mean']  / (df['te_mkt_mean'] + 1)
    df['te_iqr']        = df['te_q90'] - df['te_q10']
    df['te_seas_mkt']   = df['te_mktmo_mean'] / (df['te_mkt_mean'] + 1)
    df['te_how_prem']   = df['te_how_mean'] - df['te_mkt_mean']
    df['te_wk_prem']    = df['te_wk_mean']  - df['te_mkt_mean']
    return df

train = add_te(train)
test  = add_te(test)
print(f"Target encoding done.  G_MEAN={G_MEAN:.2f}  G_STD={G_STD:.2f}")

# =============================================================================
# 5. CORE WEATHER + CALENDAR FEATURES
#    Weather NaN filled with training-set medians (ref=train for both).
#    No test-set statistics are used anywhere.
# =============================================================================
WEATHER_COLS = [
    'global_horizontal_irradiance', 'diffuse_horizontal_irradiance',
    'direct_normal_irradiance',     'cloud_cover_total', 'cloud_cover_low',
    'cloud_cover_mid',              'cloud_cover_high',  'air_temperature_2m',
    'apparent_temperature_2m',      'dew_point_temperature_2m',
    'wet_bulb_temperature_2m',      'surface_pressure',  'relative_humidity_2m',
    'freezing_level_height',        'convective_available_potential_energy',
    'lifted_index',                 'convective_inhibition', 'wind_speed_80m',
    'wind_direction_80m',           'wind_gust_speed_10m', 'wind_speed_10m',
    'precipitation_amount',         'visibility',
]

def make_features(df, ref):
    """Build all derived features.  ref=train ensures NaN fill uses train stats."""
    df = df.copy()
    for col in WEATHER_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(ref[col].median())

    h, d, mo, doy = df['hour'], df['dow'], df['month'], df['doy']

    # Calendar flags
    df['is_peak']         = ((h >= 7)  & (h <= 22) & (d < 5)).astype(int)
    df['is_weekend']      = (d >= 5).astype(int)
    df['is_night']        = ((h >= 22) | (h <= 5)).astype(int)
    df['is_morning_ramp'] = ((h >= 6)  & (h <= 9)).astype(int)
    df['is_evening_ramp'] = ((h >= 17) & (h <= 21)).astype(int)
    df['is_monday']       = (d == 0).astype(int)
    df['is_friday']       = (d == 4).astype(int)
    df['season']          = mo.map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})

    # Cyclic encodings
    for val, period, name in [(h, 24, 'hour'), (d, 7, 'dow'), (mo, 12, 'month'), (doy, 365, 'doy')]:
        df[f'{name}_sin'] = np.sin(2 * np.pi * val / period)
        df[f'{name}_cos'] = np.cos(2 * np.pi * val / period)
    df['how_sin'] = np.sin(2 * np.pi * df['hour_of_week'] / 168)
    df['how_cos'] = np.cos(2 * np.pi * df['hour_of_week'] / 168)
    df['q_sin']   = np.sin(2 * np.pi * doy / 91.25)
    df['q_cos']   = np.cos(2 * np.pi * doy / 91.25)

    # Solar
    ghi   = df['global_horizontal_irradiance'].clip(lower=0)
    dhi   = df['diffuse_horizontal_irradiance'].clip(lower=0)
    dni   = df['direct_normal_irradiance'].clip(lower=0)
    sf    = df['solar_forecast'].clip(lower=0)
    cctot = df['cloud_cover_total'].fillna(0)
    cclo  = df['cloud_cover_low'].fillna(0)
    df['solar_total']        = ghi + dhi
    df['solar_surprise']     = df['solar_total'] - sf
    df['solar_surprise_pct'] = df['solar_surprise'] / (sf + 1)
    df['solar_is_zero']      = (ghi < 1).astype(int)
    df['effective_solar']    = ghi * (1 - cctot / 100).clip(0, 1)
    df['dni_dhi_ratio']      = (dni / (dhi + 1)).clip(0, 10)
    df['solar_cloud_block']  = ghi * cclo / 100

    # Wind
    ws80 = df['wind_speed_80m'].fillna(0).clip(0, 35)
    ws10 = df['wind_speed_10m'].fillna(0).clip(0, 30)
    wg10 = df['wind_gust_speed_10m'].fillna(0).clip(0, 40)
    wd   = df['wind_direction_80m'].fillna(180)
    wfc  = df['wind_forecast'].clip(lower=0)
    sp   = df['surface_pressure'].fillna(1013)
    df['wind_pow']      = ws80 ** 3
    df['wind_pow_sq']   = ws80 ** 2
    df['wind_surprise'] = ws80 - wfc
    df['wind_shear']    = (ws80 - ws10).clip(lower=0)
    df['gust_ratio']    = wg10 / (ws80 + 0.1)
    df['wind_dir_sin']  = np.sin(np.deg2rad(wd))
    df['wind_dir_cos']  = np.cos(np.deg2rad(wd))
    df['storm_index']   = ws80 * (1013 - sp).clip(lower=0)

    # Temperature / load / balance
    T   = df['air_temperature_2m'].fillna(15)
    Tdp = df['dew_point_temperature_2m'].fillna(5)
    Twb = df['wet_bulb_temperature_2m'].fillna(10)
    lfc = df['load_forecast']
    rfc = sf + wfc
    bal = lfc - rfc
    df['hdd']          = np.maximum(0, 18 - T)
    df['cdd']          = np.maximum(0, T - 22)
    df['dp_spread']    = T - Tdp
    df['wet_bulb']     = Twb
    df['T_sq']         = T ** 2
    df['temp_load']    = T * lfc
    df['load_log']     = np.log1p(lfc)
    df['balance']      = bal
    df['balance_pct']  = bal / (lfc + 1)
    df['supply_pct']   = rfc / (lfc + 1)
    df['balance_log']  = np.sign(bal) * np.log1p(bal.abs())
    df['balance_peak'] = bal * df['is_peak']
    df['load_solar_r'] = lfc / (sf + 1)
    df['load_wind_r']  = lfc / (wfc + 1)
    df['ren_penetr']   = rfc / (lfc + 1)

    # Atmospheric
    cape = df['convective_available_potential_energy'].fillna(0).clip(lower=0)
    cin  = df['convective_inhibition'].fillna(0)
    df['rh']        = df['relative_humidity_2m'].fillna(70)
    df['pressure']  = sp
    df['cape']      = cape
    df['conv_risk'] = cape / (cin.abs() + 1)
    df['li']        = df['lifted_index'].fillna(0)
    df['cloud_w']   = (0.5 * cclo
                       + 0.3 * df['cloud_cover_mid'].fillna(0)
                       + 0.2 * df['cloud_cover_high'].fillna(0))
    df['precip']     = df['precipitation_amount'].fillna(0).clip(0, 50)
    df['visibility'] = df['visibility'].fillna(20000).clip(0, 50000)

    # TE-interaction features
    df['bal_vs_te']        = df['balance']        - df['te_hdow_mean']
    df['solar_x_te']       = df['solar_surprise'] * df['te_hdow_mean']
    df['wind_x_te']        = df['wind_surprise']  * df['te_hdow_mean']
    df['te_range_x_bal']   = df['te_iqr']         * df['balance_pct']
    df['solar_vs_day_max'] = ghi / (df['day_solar_max'] + 1)
    df['bal_vs_day_avg']   = bal - df['day_balance'] / 24
    df['renew_vs_day_avg'] = df['ren_penetr'] - df['day_renew_pct']
    df['temp_vs_day_avg']  = T - df['day_temp_avg']
    return df

train_fe = make_features(train, ref=train)
test_fe  = make_features(test,  ref=train)

# =============================================================================
# 6. CROSS-MARKET FEATURES
#    Computed per delivery_start within each dataframe (no cross-set mixing).
# =============================================================================
def add_cross_market_features(df):
    cross = df.groupby('delivery_start').agg(
        xmkt_load_mean   = ('load_forecast', 'mean'),
        xmkt_solar_mean  = ('solar_forecast', 'mean'),
        xmkt_wind_mean   = ('wind_forecast',  'mean'),
        xmkt_balance_mean= ('balance',        'mean'),
        xmkt_renew_mean  = ('ren_penetr',     'mean'),
    ).reset_index()
    df = df.merge(cross, on='delivery_start', how='left')
    df['load_vs_xmkt']    = df['load_forecast'] - df['xmkt_load_mean']
    df['balance_vs_xmkt'] = df['balance']       - df['xmkt_balance_mean']
    df['renew_vs_xmkt']   = df['ren_penetr']    - df['xmkt_renew_mean']
    return df

# =============================================================================
# 7. INTRADAY FEATURES
#    3-hour rolling within (date, market) — no future-hour leakage within a day.
# =============================================================================
def add_intraday_features(df):
    df = df.copy()
    df['_date'] = df['delivery_start'].dt.date
    df_s = df.sort_values(['market', 'delivery_start'])
    for col in ['load_forecast', 'solar_forecast', 'wind_forecast', 'balance']:
        m_ = df_s.groupby(['market', '_date'])[col].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        s_ = df_s.groupby(['market', '_date'])[col].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        df.loc[df_s.index, f'{col}_3h_mean'] = m_.values
        df.loc[df_s.index, f'{col}_3h_std']  = s_.values
    df['hour_bal_rank']   = df.groupby(['_date', 'market'])['balance'].rank(pct=True)
    df['hour_solar_rank'] = df.groupby(['_date', 'market'])['effective_solar'].rank(pct=True)
    df['hour_load_rank']  = df.groupby(['_date', 'market'])['load_forecast'].rank(pct=True)
    return df.drop('_date', axis=1)

train_fe = add_cross_market_features(train_fe)
test_fe  = add_cross_market_features(test_fe)
train_fe = add_intraday_features(train_fe)
test_fe  = add_intraday_features(test_fe)

# =============================================================================
# 8. 1-YEAR LAG FEATURES
#    lag_src built from training data only.
#    Offsets 364-367 days: for Sep-Nov 2025 test, lags come from Sep-Nov 2024
#    which is in the training CSV.  NaN (no lag available) filled with TE mean.
# =============================================================================
print("Building 1-year lag features...")
lag_src = train[['market', 'delivery_start', 'target']].copy()

lag_364 = lag_src.copy()
lag_364['delivery_start'] += pd.Timedelta(days=364)
lag_364 = lag_364.rename(columns={'target': 'lag1y_364'}).drop_duplicates(['market', 'delivery_start'])

lag_365 = lag_src.copy()
lag_365['delivery_start'] += pd.Timedelta(days=365)
lag_365 = lag_365.rename(columns={'target': 'lag1y_365'}).drop_duplicates(['market', 'delivery_start'])

def add_exact_lags(df, te_col='te_hdow_mean'):
    df = df.merge(lag_364, on=['market', 'delivery_start'], how='left')
    df = df.merge(lag_365, on=['market', 'delivery_start'], how='left')
    df['lag1y_available']  = (~df['lag1y_364'].isna()).astype(np.float32)
    df['lag1y_364']        = df['lag1y_364'].fillna(df[te_col])
    df['lag1y_365']        = df['lag1y_365'].fillna(df[te_col])
    df['lag1y_avg']        = 0.5 * (df['lag1y_364'] + df['lag1y_365'])
    df['lag1y_vs_te']      = df['lag1y_avg']  - df[te_col]
    df['lag1y_364_vs_te']  = df['lag1y_364']  - df[te_col]
    df['lag1y_365_vs_te']  = df['lag1y_365']  - df[te_col]
    return df

OFFSETS = list(range(361, 368))
lag_7_dfs = []
for offset in OFFSETS:
    lo = lag_src.copy()
    lo['delivery_start'] += pd.Timedelta(days=offset)
    lo = lo.rename(columns={'target': f'_lag{offset}'}).drop_duplicates(['market', 'delivery_start'])
    lag_7_dfs.append(lo)

def add_7day_window(df, te_col='te_hdow_mean'):
    df = df.copy()
    for lo, offset in zip(lag_7_dfs, OFFSETS):
        df = df.merge(lo, on=['market', 'delivery_start'], how='left')
    cols = [f'_lag{o}' for o in OFFSETS]
    mat = df[cols].values.astype(float)
    mat_filled = np.where(np.isnan(mat), df[te_col].values[:, None], mat)
    df['lag1y_7avg']        = mat_filled.mean(axis=1)
    df['lag1y_7std']        = mat_filled.std(axis=1)
    df['lag1y_7min']        = mat_filled.min(axis=1)
    df['lag1y_7max']        = mat_filled.max(axis=1)
    df['lag1y_7range']      = df['lag1y_7max'] - df['lag1y_7min']
    df['lag1y_7vs_te']      = df['lag1y_7avg'] - df[te_col]
    df['lag7std_x_bal']     = df['lag1y_7std']   * df['balance']
    df['lag7std_x_balpct']  = df['lag1y_7std']   * df['balance_pct']
    df['lag7range_x_peak']  = df['lag1y_7range'] * df['is_peak'].astype(float)
    return df.drop(columns=cols)

lag_m_agg = (
    train.assign(_yr=train['delivery_start'].dt.year)
    .groupby(['_yr', 'month', 'market', 'hour_of_week'])['target']
    .agg(lag1y_mhow_mean='mean', lag1y_mhow_std='std')
    .reset_index()
)
lag_m_agg['_yr'] = lag_m_agg['_yr'] + 1  # shift forward 1 year for merge

def add_monthly_lags(df, te_col='te_hdow_mean'):
    df = df.copy()
    df['_yr'] = df['delivery_start'].dt.year
    df = df.merge(lag_m_agg, on=['_yr', 'month', 'market', 'hour_of_week'], how='left').drop('_yr', axis=1)
    df['lag1y_mhow_mean']  = df['lag1y_mhow_mean'].fillna(df[te_col])
    df['lag1y_mhow_std']   = df['lag1y_mhow_std'].fillna(G_STD)
    df['lag1y_mhow_vs_te'] = df['lag1y_mhow_mean'] - df[te_col]
    df['lag1y_mhow_ratio'] = df['lag1y_mhow_mean'] / (df[te_col].abs() + 1)
    return df

def add_lag_interactions(df):
    df  = df.copy()
    lv  = df['lag1y_vs_te'];   la = df['lag1y_avg']
    lm  = df['lag1y_mhow_vs_te']
    bal = df['balance'];       ss = df['solar_surprise']; ws = df['wind_surprise']
    bp  = df['balance_pct'];   pk = df['is_peak'].astype(float)
    df['lagvte_x_bal']       = lv * bal
    df['lagvte_x_solar']     = lv * ss
    df['lagvte_x_wind']      = lv * ws
    df['lagvte_x_peak']      = lv * pk
    df['lagvte_x_balpct']    = lv * bp
    df['lagmhow_x_bal']      = lm * bal
    df['lagavg_x_bal']       = la * bal
    df['lagavg_x_balpct']    = la * bp
    df['lagavg_x_solar']     = la * ss
    df['lagavg_x_wind']      = la * ws
    df['lag_avail_x_lag']    = df['lag1y_available'] * la
    df['lag_avail_x_lagvte'] = df['lag1y_available'] * lv
    return df

train_fe = add_exact_lags(train_fe);    test_fe = add_exact_lags(test_fe)
train_fe = add_7day_window(train_fe);   test_fe = add_7day_window(test_fe)
train_fe = add_monthly_lags(train_fe);  test_fe = add_monthly_lags(test_fe)
train_fe = add_lag_interactions(train_fe); test_fe = add_lag_interactions(test_fe)

# =============================================================================
# 9. FEATURE MATRIX + HOLDOUT SPLIT
# =============================================================================
train_fe = pd.get_dummies(train_fe, columns=['market'], prefix='mkt')
test_fe  = pd.get_dummies(test_fe,  columns=['market'], prefix='mkt')

# Columns excluded from feature matrix
DROP = ['id', 'target', 'delivery_start', 'delivery_end',
        'hour', 'dow', 'month', 'doy', 'hour_of_week', 'week']

feature_cols = [c for c in train_fe.columns
                if c not in DROP
                and c in test_fe.columns
                and train_fe[c].dtype != 'object']

# Sort chronologically, fill NaN with training-set medians
train_fe = train_fe.sort_values('delivery_start').reset_index(drop=True)
for col in feature_cols:
    med = train_fe[col].median()
    train_fe[col] = train_fe[col].fillna(med)
    test_fe[col]  = test_fe[col].fillna(med if not np.isnan(med) else 0.0)

X      = train_fe[feature_cols].values.astype(np.float32)
y      = train_fe['target'].values.astype(np.float32)
X_test = test_fe[feature_cols].values.astype(np.float32)

# Holdout windows (strictly out of model training)
mask_holdout1 = ((train_fe['month'].isin([9, 10, 11])) & (train_fe['year'] == 2023)).values
mask_holdout2 = ((train_fe['month'].isin([9, 10, 11])) & (train_fe['year'] == 2024)).values
mask_train    = ~mask_holdout1 & ~mask_holdout2
tr_idx  = np.where(mask_train)[0]
va1_idx = np.where(mask_holdout1)[0]
va2_idx = np.where(mask_holdout2)[0]

# Winsorisation bounds from training split only
y_lo, y_hi = np.percentile(y[tr_idx], [0.5, 99.5])
y_wins = np.clip(y, y_lo, y_hi)

# Exponential time-decay sample weights (half-life = 365 days)
days = (train_fe['delivery_start'] - train_fe['delivery_start'].min()).dt.days.values
sw   = np.exp(days * np.log(2) / 365)
sw  /= sw.mean()

# Per-market feature indices (no market dummies)
pm_cols = [c for c in feature_cols if not c.startswith('mkt_')]
pm_idx  = [feature_cols.index(c) for c in pm_cols]

print(f"\nFeature count: {len(feature_cols)}  ({len(pm_cols)} per-market + {len(feature_cols)-len(pm_cols)} market dummies)")
print(f"Train: {mask_train.sum():,}  Hold-2023: {mask_holdout1.sum():,}  Hold-2024: {mask_holdout2.sum():,}")

# =============================================================================
# 10. REGIME CLUSTERING
#     Scaler and KMeans fitted on tr_idx only — applied via transform elsewhere.
# =============================================================================
DAY_RF   = [c for c in ['day_balance', 'day_renew_pct', 'day_temp_avg',
                         'day_hdd', 'day_cdd', 'day_solar_sum',
                         'day_wind_sum', 'day_load_sum'] if c in feature_cols]
fidx_r   = [feature_cols.index(f) for f in DAY_RF]
scaler_r = StandardScaler()
X_reg_tr = scaler_r.fit_transform(X[tr_idx][:, fidx_r])
km = KMeans(n_clusters=8, random_state=KMEANS_SEED, n_init=10)
km.fit(X_reg_tr)

regime_col          = np.zeros(len(X), dtype=np.float32)
regime_col[tr_idx]  = km.predict(X_reg_tr).astype(np.float32)
regime_col[va1_idx] = km.predict(scaler_r.transform(X[va1_idx][:, fidx_r])).astype(np.float32)
regime_col[va2_idx] = km.predict(scaler_r.transform(X[va2_idx][:, fidx_r])).astype(np.float32)
regime_te           = km.predict(scaler_r.transform(X_test[:, fidx_r])).astype(np.float32)

X      = np.column_stack([X, regime_col])
X_test = np.column_stack([X_test, regime_te])
pm_idx = pm_idx + [X.shape[1] - 1]
print(f"Regime clusters: 8   Total features: {X.shape[1]}")

# =============================================================================
# 11. MODEL TRAINING HELPERS
# =============================================================================
def train_lgb(cfg, Xtr, ytr, sw_tr, Xes, yes):
    m = lgb.LGBMRegressor(**cfg)
    m.fit(Xtr, ytr, sample_weight=sw_tr,
          eval_set=[(Xes, yes)],
          callbacks=[lgb.early_stopping(80, verbose=False)])
    return m

def train_lgb_dart(cfg, Xtr, ytr, sw_tr):
    """DART does not support early stopping — fixed n_estimators=1500."""
    m = lgb.LGBMRegressor(**cfg)
    m.fit(Xtr, ytr, sample_weight=sw_tr)
    return m

def train_xgb(cfg, Xtr, ytr, sw_tr, Xes, yes):
    m = xgb.XGBRegressor(**cfg)
    m.fit(Xtr, ytr, sample_weight=sw_tr, eval_set=[(Xes, yes)], verbose=False)
    return m

# =============================================================================
# 12. MODEL CONFIGS
# =============================================================================
# Per-market LGB (deep tree + regularised blend)
LGB_DEEP = dict(n_estimators=4000, learning_rate=0.02,  num_leaves=127,
                min_child_samples=15, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=5, reg_alpha=0.05, reg_lambda=0.1,
                verbose=-1, n_jobs=-1, random_state=PM_LGB_SEED)
LGB_REG  = dict(n_estimators=3000, learning_rate=0.03,  num_leaves=63,
                min_child_samples=25, feature_fraction=0.75, bagging_fraction=0.75,
                bagging_freq=5, reg_alpha=0.4, reg_lambda=0.8,
                verbose=-1, n_jobs=-1, random_state=PM_LGB_SEED)

# Per-market XGB
XGB_PM = dict(n_estimators=3000, learning_rate=0.025, max_depth=7,
              min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=80,
              tree_method='hist', verbosity=0, random_state=PM_XGB_SEED)

# Global XGB base configs (random_state injected per seed in loop)
XGB_D7_BASE = dict(n_estimators=3000, learning_rate=0.025, max_depth=7,
                   min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=1.0, early_stopping_rounds=80,
                   tree_method='hist', verbosity=0)
XGB_D8_BASE = dict(n_estimators=3000, learning_rate=0.025, max_depth=8,
                   min_child_weight=3, subsample=0.8, colsample_bytree=0.75,
                   reg_alpha=0.15, reg_lambda=1.5, early_stopping_rounds=80,
                   tree_method='hist', verbosity=0)

# Global LGB
LGB_GLOBAL_D127 = dict(n_estimators=4000, learning_rate=0.02, num_leaves=127,
                        min_child_samples=15, feature_fraction=0.8, bagging_fraction=0.8,
                        bagging_freq=5, reg_alpha=0.05, reg_lambda=0.1,
                        verbose=-1, n_jobs=-1, random_state=LGB_D127_SEED)
LGB_GLOBAL_D255 = dict(n_estimators=4000, learning_rate=0.02, num_leaves=255,
                        min_child_samples=20, feature_fraction=0.75, bagging_fraction=0.75,
                        bagging_freq=5, reg_alpha=0.1, reg_lambda=0.3,
                        verbose=-1, n_jobs=-1, random_state=LGB_D255_SEED)

# Global LGB DART base config (random_state injected per seed in loop)
DART_BASE = dict(boosting_type='dart', n_estimators=1500, learning_rate=0.025,
                 num_leaves=127, min_child_samples=15, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=5, drop_rate=0.1,
                 skip_drop=0.5, max_drop=50, reg_alpha=0.05, reg_lambda=0.1,
                 verbose=-1, n_jobs=-1)

# =============================================================================
# 13. PER-MARKET LGB
# =============================================================================
print("\n" + "="*60 + "\nPER-MARKET LGB\n" + "="*60)

oof_lgb_deep  = np.zeros(len(X), np.float32)
oof_lgb_reg   = np.zeros(len(X), np.float32)
test_lgb_deep = np.zeros(len(X_test), np.float32)
test_lgb_reg  = np.zeros(len(X_test), np.float32)

print(f"  {'Market':<12} {'Deep-23':>8} {'Reg-23':>7} {'Deep-24':>8} {'Reg-24':>7}")
print("  " + "-"*45)
for mkt in MARKETS:
    mc    = f'mkt_{mkt}'
    tr_m  = np.where(mask_train    & (train_fe[mc] == True).values)[0]
    va1_m = np.where(mask_holdout1 & (train_fe[mc] == True).values)[0]
    va2_m = np.where(mask_holdout2 & (train_fe[mc] == True).values)[0]
    te_m  = np.where(test_fe[mc].values)[0]
    n_es  = max(int(len(tr_m) * 0.1), 100)
    Xm_tr = X[tr_m][:, pm_idx]
    Xm_te = X_test[te_m][:, pm_idx]
    m1 = train_lgb(LGB_DEEP, Xm_tr[:-n_es], y_wins[tr_m[:-n_es]], sw[tr_m[:-n_es]], Xm_tr[-n_es:], y_wins[tr_m[-n_es:]])
    m2 = train_lgb(LGB_REG,  Xm_tr[:-n_es], y_wins[tr_m[:-n_es]], sw[tr_m[:-n_es]], Xm_tr[-n_es:], y_wins[tr_m[-n_es:]])
    for va_m in [va1_m, va2_m]:
        oof_lgb_deep[va_m] = m1.predict(X[va_m][:, pm_idx])
        oof_lgb_reg[va_m]  = m2.predict(X[va_m][:, pm_idx])
    test_lgb_deep[te_m] = m1.predict(Xm_te)
    test_lgb_reg[te_m]  = m2.predict(Xm_te)
    r1d = rmse(y[va1_m], oof_lgb_deep[va1_m]); r1r = rmse(y[va1_m], oof_lgb_reg[va1_m])
    r2d = rmse(y[va2_m], oof_lgb_deep[va2_m]); r2r = rmse(y[va2_m], oof_lgb_reg[va2_m])
    print(f"  {mkt:<12} {r1d:8.4f} {r1r:7.4f} {r2d:8.4f} {r2r:7.4f}")
    del m1, m2; gc.collect()

oof_lgb_bl  = 0.5 * oof_lgb_deep + 0.5 * oof_lgb_reg
test_lgb_bl = 0.5 * test_lgb_deep + 0.5 * test_lgb_reg
r1 = rmse(y[va1_idx], oof_lgb_bl[va1_idx])
r2 = rmse(y[va2_idx], oof_lgb_bl[va2_idx])
print(f"  pm_lgb_blend: 2023={r1:.4f}  2024={r2:.4f}  dual={0.5*(r1+r2):.4f}")

# =============================================================================
# 14. PER-MARKET XGB
# =============================================================================
print("\n" + "="*60 + "\nPER-MARKET XGB\n" + "="*60)

oof_xgb_pm  = np.zeros(len(X), np.float32)
test_xgb_pm = np.zeros(len(X_test), np.float32)

for mkt in MARKETS:
    mc    = f'mkt_{mkt}'
    tr_m  = np.where(mask_train    & (train_fe[mc] == True).values)[0]
    va1_m = np.where(mask_holdout1 & (train_fe[mc] == True).values)[0]
    va2_m = np.where(mask_holdout2 & (train_fe[mc] == True).values)[0]
    te_m  = np.where(test_fe[mc].values)[0]
    n_es  = max(int(len(tr_m) * 0.1), 100)
    Xm_tr = X[tr_m][:, pm_idx]
    Xm_te = X_test[te_m][:, pm_idx]
    m = train_xgb(XGB_PM, Xm_tr[:-n_es], y_wins[tr_m[:-n_es]], sw[tr_m[:-n_es]], Xm_tr[-n_es:], y_wins[tr_m[-n_es:]])
    for va_m in [va1_m, va2_m]:
        oof_xgb_pm[va_m] = m.predict(X[va_m][:, pm_idx])
    test_xgb_pm[te_m] = m.predict(Xm_te)
    r1 = rmse(y[va1_m], oof_xgb_pm[va1_m]); r2 = rmse(y[va2_m], oof_xgb_pm[va2_m])
    print(f"  {mkt}: 2023={r1:.4f}  2024={r2:.4f}  iters={m.best_iteration}")
    del m; gc.collect()

r1 = rmse(y[va1_idx], oof_xgb_pm[va1_idx])
r2 = rmse(y[va2_idx], oof_xgb_pm[va2_idx])
print(f"  pm_xgb: 2023={r1:.4f}  2024={r2:.4f}  dual={0.5*(r1+r2):.4f}")

# =============================================================================
# 15. GLOBAL XGB depth=7  (7 seeds)
# =============================================================================
print("\n" + "="*60)
print(f"GLOBAL XGB depth=7 — seeds: {D7_SEEDS}")
print("="*60)

n_es_g = max(int(len(tr_idx) * 0.1), 500)
oof_xgb_d7_list  = []
test_xgb_d7_list = []

for seed in D7_SEEDS:
    cfg = {**XGB_D7_BASE, 'random_state': seed}
    m = train_xgb(cfg,
                  X[tr_idx[:-n_es_g]], y_wins[tr_idx[:-n_es_g]], sw[tr_idx[:-n_es_g]],
                  X[tr_idx[-n_es_g:]],  y_wins[tr_idx[-n_es_g:]])
    oof = np.zeros(len(X), np.float32)
    oof[va1_idx] = m.predict(X[va1_idx])
    oof[va2_idx] = m.predict(X[va2_idx])
    r1 = rmse(y[va1_idx], oof[va1_idx]); r2 = rmse(y[va2_idx], oof[va2_idx])
    print(f"  g_xgb_d7_s{seed:<6}: 2023={r1:.4f}  2024={r2:.4f}  iters={m.best_iteration}")
    oof_xgb_d7_list.append(oof)
    test_xgb_d7_list.append(m.predict(X_test).astype(np.float32))
    del m; gc.collect()

# =============================================================================
# 16. GLOBAL XGB depth=8  (5 seeds)
# =============================================================================
print("\n" + "="*60)
print(f"GLOBAL XGB depth=8 — seeds: {D8_SEEDS}")
print("="*60)

oof_xgb_d8_list  = []
test_xgb_d8_list = []

for seed in D8_SEEDS:
    cfg = {**XGB_D8_BASE, 'random_state': seed}
    m = train_xgb(cfg,
                  X[tr_idx[:-n_es_g]], y_wins[tr_idx[:-n_es_g]], sw[tr_idx[:-n_es_g]],
                  X[tr_idx[-n_es_g:]],  y_wins[tr_idx[-n_es_g:]])
    oof = np.zeros(len(X), np.float32)
    oof[va1_idx] = m.predict(X[va1_idx])
    oof[va2_idx] = m.predict(X[va2_idx])
    r1 = rmse(y[va1_idx], oof[va1_idx]); r2 = rmse(y[va2_idx], oof[va2_idx])
    print(f"  g_xgb_d8_s{seed:<6}: 2023={r1:.4f}  2024={r2:.4f}  iters={m.best_iteration}")
    oof_xgb_d8_list.append(oof)
    test_xgb_d8_list.append(m.predict(X_test).astype(np.float32))
    del m; gc.collect()

# =============================================================================
# 17. GLOBAL LGB  num_leaves=127 and num_leaves=255
# =============================================================================
print("\n" + "="*60 + "\nGLOBAL LGB d127 + d255\n" + "="*60)

n_es_glgb = max(int(len(tr_idx) * 0.1), 500)

lgb_d127 = train_lgb(LGB_GLOBAL_D127,
                     X[tr_idx[:-n_es_glgb]], y_wins[tr_idx[:-n_es_glgb]], sw[tr_idx[:-n_es_glgb]],
                     X[tr_idx[-n_es_glgb:]], y_wins[tr_idx[-n_es_glgb:]])
oof_lgb_d127 = np.zeros(len(X), np.float32)
oof_lgb_d127[va1_idx] = lgb_d127.predict(X[va1_idx])
oof_lgb_d127[va2_idx] = lgb_d127.predict(X[va2_idx])
test_lgb_d127 = lgb_d127.predict(X_test).astype(np.float32)
r1 = rmse(y[va1_idx], oof_lgb_d127[va1_idx]); r2 = rmse(y[va2_idx], oof_lgb_d127[va2_idx])
print(f"  g_lgb_d127: 2023={r1:.4f}  2024={r2:.4f}  iters={lgb_d127.best_iteration_}")
del lgb_d127; gc.collect()

lgb_d255 = train_lgb(LGB_GLOBAL_D255,
                     X[tr_idx[:-n_es_glgb]], y_wins[tr_idx[:-n_es_glgb]], sw[tr_idx[:-n_es_glgb]],
                     X[tr_idx[-n_es_glgb:]], y_wins[tr_idx[-n_es_glgb:]])
oof_lgb_d255 = np.zeros(len(X), np.float32)
oof_lgb_d255[va1_idx] = lgb_d255.predict(X[va1_idx])
oof_lgb_d255[va2_idx] = lgb_d255.predict(X[va2_idx])
test_lgb_d255 = lgb_d255.predict(X_test).astype(np.float32)
r1 = rmse(y[va1_idx], oof_lgb_d255[va1_idx]); r2 = rmse(y[va2_idx], oof_lgb_d255[va2_idx])
print(f"  g_lgb_d255: 2023={r1:.4f}  2024={r2:.4f}  iters={lgb_d255.best_iteration_}")
del lgb_d255; gc.collect()

# =============================================================================
# 18. GLOBAL LGB DART  (4 seeds)
# =============================================================================
print("\n" + "="*60)
print(f"GLOBAL LGB DART — seeds: {DART_SEEDS}")
print("="*60)

oof_dart_list  = []
test_dart_list = []

for seed in DART_SEEDS:
    cfg = {**DART_BASE, 'random_state': seed}
    m = train_lgb_dart(cfg, X[tr_idx], y_wins[tr_idx], sw[tr_idx])
    oof = np.zeros(len(X), np.float32)
    oof[va1_idx] = m.predict(X[va1_idx])
    oof[va2_idx] = m.predict(X[va2_idx])
    r1 = rmse(y[va1_idx], oof[va1_idx]); r2 = rmse(y[va2_idx], oof[va2_idx])
    print(f"  g_lgb_dart_s{seed:<4}: 2023={r1:.4f}  2024={r2:.4f}  n_est=1500")
    oof_dart_list.append(oof)
    test_dart_list.append(m.predict(X_test).astype(np.float32))
    del m; gc.collect()

# =============================================================================
# 19. SLSQP PER-MARKET WEIGHT OPTIMISATION
#     Objective: minimise RMSE on holdout-2024 only.
#     100 random Dirichlet starts with seed SLSQP_SEED to avoid local minima.
# =============================================================================
N_D7    = len(D7_SEEDS)
N_D8    = len(D8_SEEDS)
N_DART  = len(DART_SEEDS)
N_TOTAL = 2 + N_D7 + N_D8 + 2 + N_DART   # pm_lgb + pm_xgb + d7 + d8 + lgb2 + dart

model_names = (['pmLGB', 'pmXGB']
               + [f'D7s{s}'   for s in D7_SEEDS]
               + [f'D8s{s}'   for s in D8_SEEDS]
               + ['L127', 'L255']
               + [f'DARTs{s}' for s in DART_SEEDS])

print(f"\n{'='*70}")
print(f"SLSQP {N_TOTAL}-MODEL WEIGHT OPTIMISATION  (holdout-2024, {N_SLSQP_STARTS} starts/market)")
print(f"{'='*70}")

def optimize_weights_slsqp(preds_list, y_true, n_starts=100, seed=SLSQP_SEED):
    n   = len(preds_list)
    P   = np.column_stack([p.astype(np.float64) for p in preds_list])
    y64 = y_true.astype(np.float64)
    def loss(w): return float(np.sqrt(np.mean((P @ w - y64) ** 2)))
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    rng = np.random.RandomState(seed)
    best_val, best_w = np.inf, np.ones(n) / n
    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n))
        try:
            res = minimize(loss, w0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'ftol': 1e-12, 'maxiter': 2000})
            if res.fun < best_val:
                best_val, best_w = res.fun, res.x
        except Exception:
            pass
    best_w = np.maximum(0, best_w); best_w /= best_w.sum()
    return best_w, best_val

best_market_weights = {}

hdr = "  {:10s}  ".format("Market") + "  ".join(f"{n:>8}" for n in model_names) + "    2024    2023"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for mkt in MARKETS:
    mc    = f'mkt_{mkt}'
    va1_m = np.where(mask_holdout1 & (train_fe[mc] == True).values)[0]
    va2_m = np.where(mask_holdout2 & (train_fe[mc] == True).values)[0]
    y2    = y[va2_m]; y1 = y[va1_m]

    def preds_for(idx):
        return ([oof_lgb_bl[idx], oof_xgb_pm[idx]]
                + [o[idx] for o in oof_xgb_d7_list]
                + [o[idx] for o in oof_xgb_d8_list]
                + [oof_lgb_d127[idx], oof_lgb_d255[idx]]
                + [o[idx] for o in oof_dart_list])

    w, r2_opt = optimize_weights_slsqp(preds_for(va2_m), y2)
    best_market_weights[mkt] = w
    P1 = np.column_stack([p.astype(np.float64) for p in preds_for(va1_m)])
    r1 = rmse(y1, P1 @ w)
    wfmt = "  ".join(f"{wi:8.3f}" for wi in w)
    print(f"  {mkt:<10}  {wfmt}   {r2_opt:7.4f} {r1:7.4f}")

# Ensemble OOF for reporting
oof_ens = np.zeros(len(X), np.float32)
for mkt in MARKETS:
    mc     = f'mkt_{mkt}'
    va_all = np.where((mask_holdout1 | mask_holdout2) & (train_fe[mc] == True).values)[0]
    w = best_market_weights[mkt]
    all_oofs = ([oof_lgb_bl[va_all], oof_xgb_pm[va_all]]
                + [o[va_all] for o in oof_xgb_d7_list]
                + [o[va_all] for o in oof_xgb_d8_list]
                + [oof_lgb_d127[va_all], oof_lgb_d255[va_all]]
                + [o[va_all] for o in oof_dart_list])
    oof_ens[va_all] = sum(w[i] * all_oofs[i] for i in range(N_TOTAL))

r1_ens  = rmse(y[va1_idx], oof_ens[va1_idx])
r2_ens  = rmse(y[va2_idx], oof_ens[va2_idx])
dual_pm = 0.5 * (r1_ens + r2_ens)
print(f"\n  Ensemble: 2023={r1_ens:.4f}  2024={r2_ens:.4f}  dual50={dual_pm:.4f}")

# =============================================================================
# 20. RETRAIN ALL MODELS ON FULL TRAINING DATA
# =============================================================================
print("\n" + "="*60 + "\nRETRAIN ON ALL DATA\n" + "="*60)

final_lgb_deep = np.zeros(len(X_test), np.float32)
final_lgb_reg  = np.zeros(len(X_test), np.float32)
final_xgb_pm   = np.zeros(len(X_test), np.float32)

for mkt in MARKETS:
    mc     = f'mkt_{mkt}'
    all_m  = np.where(train_fe[mc].values)[0]
    te_m   = np.where(test_fe[mc].values)[0]
    n_es   = max(int(len(all_m) * 0.08), 100)
    Xm_all = X[all_m][:, pm_idx]
    Xm_te  = X_test[te_m][:, pm_idx]
    m1 = train_lgb(LGB_DEEP, Xm_all[:-n_es], y_wins[all_m[:-n_es]], sw[all_m[:-n_es]], Xm_all[-n_es:], y_wins[all_m[-n_es:]])
    m2 = train_lgb(LGB_REG,  Xm_all[:-n_es], y_wins[all_m[:-n_es]], sw[all_m[:-n_es]], Xm_all[-n_es:], y_wins[all_m[-n_es:]])
    m3 = train_xgb(XGB_PM,   Xm_all[:-n_es], y_wins[all_m[:-n_es]], sw[all_m[:-n_es]], Xm_all[-n_es:], y_wins[all_m[-n_es:]])
    final_lgb_deep[te_m] = m1.predict(Xm_te)
    final_lgb_reg[te_m]  = m2.predict(Xm_te)
    final_xgb_pm[te_m]   = m3.predict(Xm_te)
    print(f"  {mkt}: lgb_d={m1.best_iteration_:3d}  lgb_r={m2.best_iteration_:3d}  pmxgb={m3.best_iteration:3d}")
    del m1, m2, m3; gc.collect()

n_es_gf = max(int(len(X) * 0.08), 500)

final_xgb_d7_list = []
for seed in D7_SEEDS:
    m = train_xgb({**XGB_D7_BASE, 'random_state': seed},
                  X[:-n_es_gf], y_wins[:-n_es_gf], sw[:-n_es_gf], X[-n_es_gf:], y_wins[-n_es_gf:])
    final_xgb_d7_list.append(m.predict(X_test).astype(np.float32))
    print(f"  retrain d7_s{seed}: iters={m.best_iteration}")
    del m; gc.collect()

final_xgb_d8_list = []
for seed in D8_SEEDS:
    m = train_xgb({**XGB_D8_BASE, 'random_state': seed},
                  X[:-n_es_gf], y_wins[:-n_es_gf], sw[:-n_es_gf], X[-n_es_gf:], y_wins[-n_es_gf:])
    final_xgb_d8_list.append(m.predict(X_test).astype(np.float32))
    print(f"  retrain d8_s{seed}: iters={m.best_iteration}")
    del m; gc.collect()

lgb_d127_f = train_lgb(LGB_GLOBAL_D127,
                        X[:-n_es_gf], y_wins[:-n_es_gf], sw[:-n_es_gf], X[-n_es_gf:], y_wins[-n_es_gf:])
lgb_d255_f = train_lgb(LGB_GLOBAL_D255,
                        X[:-n_es_gf], y_wins[:-n_es_gf], sw[:-n_es_gf], X[-n_es_gf:], y_wins[-n_es_gf:])
final_lgb_d127_g = lgb_d127_f.predict(X_test).astype(np.float32)
final_lgb_d255_g = lgb_d255_f.predict(X_test).astype(np.float32)
print(f"  retrain lgb127={lgb_d127_f.best_iteration_}  lgb255={lgb_d255_f.best_iteration_}")
del lgb_d127_f, lgb_d255_f; gc.collect()

final_dart_list = []
for seed in DART_SEEDS:
    m = train_lgb_dart({**DART_BASE, 'random_state': seed}, X, y_wins, sw)
    final_dart_list.append(m.predict(X_test).astype(np.float32))
    print(f"  retrain dart_s{seed}: n_est=1500")
    del m; gc.collect()

final_lgb_bl = 0.5 * final_lgb_deep + 0.5 * final_lgb_reg

# =============================================================================
# 21. ASSEMBLE FINAL PREDICTIONS + SAVE
# =============================================================================
preds = np.zeros(len(X_test), np.float32)
for mkt in MARKETS:
    mc   = f'mkt_{mkt}'
    te_m = np.where(test_fe[mc].values)[0]
    w    = best_market_weights[mkt]
    all_finals = ([final_lgb_bl[te_m], final_xgb_pm[te_m]]
                  + [f[te_m] for f in final_xgb_d7_list]
                  + [f[te_m] for f in final_xgb_d8_list]
                  + [final_lgb_d127_g[te_m], final_lgb_d255_g[te_m]]
                  + [f[te_m] for f in final_dart_list])
    preds[te_m] = sum(w[i] * all_finals[i] for i in range(N_TOTAL))

# Clip to training-set target range
q_lo = float(np.percentile(y, 0.5)); q_hi = float(np.percentile(y, 99.5))
preds = np.clip(preds, q_lo, q_hi)

sub = pd.DataFrame({'id': test_fe['id'].values, 'target': preds.astype(np.float64)})
sub = sample[['id']].merge(sub, on='id', how='left')
sub['target'] = sub['target'].fillna(float(y.mean()))

assert len(sub) == 13098,           f"Row count mismatch: {len(sub)}"
assert sub['target'].notna().all(), "NaN values in submission"
assert (sub['target'] >= q_lo).all() and (sub['target'] <= q_hi).all(), "Predictions out of clipped range"

sub[['id', 'target']].to_csv('submission_v34.csv', index=False)

elapsed = (time.time() - t0) / 60
print(f"\n{'='*60}")
print(f"SAVED: submission_v34.csv")
print(f"Holdout RMSE: 2023={r1_ens:.4f}  2024={r2_ens:.4f}  dual50={dual_pm:.4f}")
print(f"Predictions: mean={preds.mean():.2f}  std={preds.std():.2f}  neg={100*(preds<0).mean():.1f}%")
print(f"Models: {N_D7} XGB-d7 + {N_D8} XGB-d8 + 2 LGB + {N_DART} DART + 2 pm = {N_TOTAL} total")
print(f"Runtime: {elapsed:.1f} min")
print(f"{'='*60}")
