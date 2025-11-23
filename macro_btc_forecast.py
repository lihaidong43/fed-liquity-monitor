# 文件名: macro_btc_forecast.py
# 功能: Predict BTC, Unemployment, CPI for next 3 months
# Language: English only in charts

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import json
import os
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# ------------------- 1. Load config -------------------
with open('config.json', 'r') as f:
    config = json.load(f)
BOT_TOKEN = config['TELEGRAM_TOKEN']
CHAT_ID = config['TELEGRAM_CHAT_ID']

# ------------------- 2. Font Setup (No Chinese in output) -------------------
font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_path = "SourceHanSansSC-Regular.otf"

if not os.path.exists(font_path):
    print("Downloading font for compatibility...")
    import urllib.request
    urllib.request.urlretrieve(font_url, font_path)

font = FontProperties(fname=font_path, size=12)
plt.rcParams['font.sans-serif'] = [font.get_name()]
plt.rcParams['axes.unicode_minus'] = False
try:
    fm._get_fontconfig_fonts()
except:
    pass

# ------------------- 3. Data Fetch -------------------
def fetch_fred(series_id, start='2015-01-01'):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    try:
        df = pd.read_csv(url)
        date_col = next(col for col in df.columns if 'date' in col.lower())
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)[series_id]
        df = df.resample('ME').last()
        return df.rename(series_id.replace('_', '').lower())
    except Exception as e:
        print(f"Failed {series_id}: {e}")
        return pd.Series(dtype='float64')

def fetch_btc_daily():
    btc_data = []
    start = datetime(2019, 1, 1)
    end = datetime.now()
    batch_days = 900

    current = start
    while current < end:
        batch_end = min(current + timedelta(days=batch_days - 1), end)
        start_unix = int(current.timestamp() * 1000)
        end_unix = int(batch_end.timestamp() * 1000) + 86400000
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime={start_unix}&endTime={end_unix}&limit=1000"
        try:
            data = requests.get(url, timeout=30).json()
            if data:
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['btc'] = pd.to_numeric(df['close'])
                df = df[['date', 'btc']].set_index('date')
                btc_data.append(df)
        except Exception as e:
            print(f"Batch failed: {e}")
        current = batch_end + timedelta(days=1)

    if btc_data:
        btc = pd.concat(btc_data).sort_index()
        btc = btc[~btc.index.duplicated(keep='last')]
        btc = btc.resample('ME').last()
        print(f"Total BTC records: {len(btc)} from {btc.index.min()} to {btc.index.max()}")
        return btc['btc']
    return pd.Series(dtype='float64')

print("Fetching FRED data...")
gdp      = fetch_fred('GDP')
unrate   = fetch_fred('UNRATE')
cpi      = fetch_fred('CPIAUCSL')
pce      = fetch_fred('PCEPI')
fed_bs   = fetch_fred('WALCL') / 1e6
tga      = fetch_fred('WTREGEN')
rrp      = fetch_fred('RRPONTSYD') / 1e6
sofr     = fetch_fred('SOFR')
ior      = fetch_fred('IORB')

print("Fetching BTC data...")
btc = fetch_btc_daily()

# ------------------- 4. Data Alignment -------------------
start_month = btc.index.min()
monthly = pd.DataFrame(index=pd.date_range(start_month,
                                          (datetime.now() + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d'),
                                          freq='ME'))

monthly = monthly.join(btc, how='left')
monthly = monthly.join(gdp, how='left')
monthly = monthly.join(unrate, how='left')
monthly = monthly.join(fed_bs, how='left')
monthly = monthly.join(tga, how='left')
monthly = monthly.join(rrp, how='left')
monthly = monthly.join(sofr, how='left')
monthly = monthly.join(ior, how='left')

# YoY (delayed)
cpi_yoy = cpi.pct_change(12) * 100
pce_yoy = pce.pct_change(12) * 100
yoy_start = max(cpi_yoy.first_valid_index(), pce_yoy.first_valid_index())
monthly = monthly.loc[yoy_start:]
monthly['cpi_yoy'] = cpi_yoy.loc[yoy_start:]
monthly['pce_yoy'] = pce_yoy.loc[yoy_start:]

monthly = monthly.rename(columns={
    'walcl': 'fed_bs',
    'wtregen': 'tga',
    'rrpontsyd': 'rrp'
})

monthly['liquidity'] = monthly['fed_bs'] - monthly['rrp'] - monthly['tga']/1000
monthly['sofr_spread'] = monthly['sofr'] - 4.0
monthly['sofr_ior_spread'] = monthly['sofr'] - monthly['iorb']
monthly['liquidity_lag1'] = monthly['liquidity'].shift(1)
monthly['liquidity_lag3'] = monthly['liquidity'].shift(3)

monthly = monthly.ffill().bfill()
df = monthly.dropna()

print(f"Final valid rows: {len(df)} from {df.index.min()} to {df.index.max()}")
if len(df) < 6:
    raise ValueError(f"Insufficient data: {len(df)} rows, need at least 6 months.")

# ------------------- 5. Model Training -------------------
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

y_cols = ['btc', 'unrate', 'cpi_yoy']
models = {}
n_test = min(3, max(1, len(df) // 4))

for col in y_cols:
    print(f"\nTraining model for: {col}")
    X = df.drop(columns=[c for c in y_cols if c in df.columns] + ['gdp', 'pce_yoy'])
    y = df[col]
    train = df.iloc[:-n_test]
    test  = df.iloc[-n_test:]

    model = XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(train[X.columns], train[col])

    if len(test) > 0:
        pred = model.predict(test[X.columns])
        mae = mean_absolute_error(test[col], pred)
        print(f"MAE: {mae:.3f}")
    else:
        mae = np.nan
        print("Test set empty")

    models[col] = {'model': model, 'features': X.columns, 'mae': mae}

# ------------------- 6. Forecast Next 3 Months -------------------
last_month_end = df.index[-1]
future_dates = pd.date_range(start=last_month_end + pd.offsets.MonthEnd(1),
                             periods=3,
                             freq='ME')
future = pd.DataFrame(index=future_dates)
last = df.iloc[-1]
for col in ['fed_bs','tga','rrp','sofr','iorb','liquidity','sofr_spread','sofr_ior_spread','liquidity_lag1','liquidity_lag3']:
    future[col] = last[col]

forecast = {}
for col in y_cols:
    pred = models[col]['model'].predict(future[models[col]['features']])
    forecast[col] = pred

forecast_df = pd.DataFrame(forecast, index=future_dates).round(2)
print("\nNext 3 Months Forecast:")
print(forecast_df)

# ------------------- 7. Plot (English) -------------------
plt.figure(figsize=(15, 10))

plt.subplot(3,1,1)
plt.plot(df.index[-24:], df['btc'].iloc[-24:], label='Historical', marker='o')
plt.plot(forecast_df.index, forecast_df['btc'], label='Forecast', marker='s', color='red')
plt.title('BTC Price Forecast (Monthly)')
plt.ylabel('USD')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(3,1,2)
plt.plot(df.index[-24:], df['unrate'].iloc[-24:], label='Historical', marker='o')
plt.plot(forecast_df.index, forecast_df['unrate'], label='Forecast', marker='s', color='red')
plt.title('Unemployment Rate Forecast')
plt.ylabel('%')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(3,1,3)
plt.plot(df.index[-24:], df['cpi_yoy'].iloc[-24:], label='Historical', marker='o')
plt.plot(forecast_df.index, forecast_df['cpi_yoy'], label='Forecast', marker='s', color='red')
plt.title('CPI Inflation Forecast (YoY)')
plt.ylabel('%')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('charts/macro_btc_forecast.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------- 8. Telegram -------------------
def send_telegram(message, image_path):
    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                  data={'chat_id': CHAT_ID, 'text': message})
    with open(image_path, 'rb') as f:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                      data={'chat_id': CHAT_ID}, files={'photo': f})

msg = f"Macro + BTC Forecast Update\nPeriod: {forecast_df.index[0].strftime('%Y-%m')} ~ {forecast_df.index[-1].strftime('%Y-%m')}\n\n{'Date':<12} {'BTC':<8} {'Unemp':<6} {'CPI':<6}\n" + "\n".join([f"{d.strftime('%Y-%m'):<12} {b:<8} {u:<6} {c:<6}" for d,b,u,c in zip(forecast_df.index, forecast_df['btc'], forecast_df['unrate'], forecast_df['cpi_yoy'])])
send_telegram(msg, 'charts/macro_btc_forecast.png')
print("Forecast completed, chart sent to Telegram!")