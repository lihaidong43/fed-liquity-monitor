# 文件名: liquidity_monitor_btc_forever_visible.py
# BTC 曲线永不消失！强制显示！

import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
import time

# ==================== 配置 ====================
with open("config.json") as f:
    config = json.load(f)
FRED_KEY = config["FRED_API_KEY"]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="2025 流动性监控", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: none !important; padding: 1rem !important;}
    .main > div {max-width: 90% !important; margin: auto !important;}
    header, #MainMenu, footer {visibility: hidden !important;}
</style>
""", unsafe_allow_html=True)

st.title("2025 美元流动性 & BTC 实时监控系统")

# ==================== 数据获取 ====================
@st.cache_data(ttl=3600)
def fred_api(sid):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {'series_id': sid, 'api_key': FRED_KEY, 'file_type': 'json', 'limit': 10000}
    try:
        data = requests.get(url, params=params, timeout=20).json()['observations']
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        series = df.set_index('date')['value']
        series.to_csv(f"{DATA_DIR}/{sid}.csv")
        return series
    except:
        cache_file = f"{DATA_DIR}/{sid}.csv"
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=['date'], index_col='date').squeeze()
        return pd.Series()

@st.cache_data(ttl=86400)
def get_btc_full():
    try:
        start_ts = 1502928000000
        end_ts = int(datetime.now().timestamp() * 1000)
        all_data = []
        current = start_ts
        while current < end_ts:
            url = "https://api.binance.com/api/v3/klines"
            params = {'symbol': 'BTCUSDT', 'interval': '1d', 'startTime': current, 'limit': 1000}
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
            if not data or isinstance(data, dict):
                time.sleep(1)
                continue
            all_data.extend(data)
            current = data[-1][0] + 86400000
            time.sleep(0.12)
        df = pd.DataFrame(all_data, columns=['ts','o','h','l','c','v','ct','qv','t','tb','tq','i'])
        df['date'] = pd.to_datetime(df['ts'], unit='ms')
        df['btc'] = pd.to_numeric(df['c'])
        series = df.set_index('date')['btc']
        series.to_csv(f"{DATA_DIR}/BTC.csv")
        return series
    except Exception as e:
        st.error(f"BTC 数据加载失败: {e}")
        cache_file = f"{DATA_DIR}/BTC.csv"
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=['date'], index_col='date').squeeze()
        return pd.Series(dtype=float)  # 永远不返回 None

# ==================== 图表生成（BTC 永久强制显示） ====================
def make_chart(start_year):
    start = f"{start_year}-01-01"
    dates = pd.date_range(start, datetime.now(), freq='D')

    tga_raw   = fred_api('WTREGEN')
    rrp_raw   = fred_api('RRPONTSYD')
    walcl_raw = fred_api('WALCL')
    sofr      = fred_api('SOFR')
    btc_full  = get_btc_full()

    # 正确换算
    tga   = tga_raw   / 1_000_000
    rrp   = rrp_raw   / 1_000
    walcl = walcl_raw / 1_000_000

    df = pd.DataFrame(index=dates)
    df['TGA']     = tga.reindex(dates)
    df['RRP']     = rrp.reindex(dates)
    df['WALCL']   = walcl.reindex(dates)
    df['SOFR']    = sofr.reindex(dates)
    df['BTC']     = btc_full.reindex(dates)  # 关键：这里不再 .ffill()

    df['Liquidity']   = df['WALCL'] - df['RRP'] - df['TGA']
    df['SOFR_Spread'] = df['SOFR'] * 100 - 400

    titles = ["BTC 价格", "美元流动性", "美联储资产负债表 WALCL", "TGA", "RRP", "SOFR", "SOFR Spread"]
    colors = ['#FF6B00', '#0066FF', '#000000', '#00AA00', '#8B4513', '#AA00AA', '#FF3333']

    fig = make_subplots(rows=7, cols=1, subplot_titles=titles, vertical_spacing=0.08)

    # 关键修复：BTC 强制取非空数据并绘制
    btc_clean = df['BTC'].dropna()
    if len(btc_clean) > 0:
        fig.add_trace(go.Scatter(
            x=btc_clean.index,
            y=btc_clean.values,
            name="BTC 价格",
            line=dict(color="#FF6B00", width=5),
            hovertemplate="日期: %{x}<br>价格: $%{y:,.0f}<extra></extra>"
        ), row=1, col=1)
    fig.update_yaxes(title_text="BTC Price (USD)", row=1, col=1, tickformat=",")

    # 其他指标（真实值）
    data_list = [df['Liquidity'], df['WALCL'], df['TGA'], df['RRP'], df['SOFR'], df['SOFR_Spread']]
    yaxis_titles = ["万亿美元", "万亿美元", "万亿美元", "万亿美元", "%", "基点"]
    for i, series in enumerate(data_list):
        clean = series.dropna()
        if len(clean) == 0: continue
        fig.add_trace(go.Scatter(x=clean.index, y=clean.values, name=titles[i+1], 
                                line=dict(color=colors[i+1], width=3)),
                      row=i+2, col=1)
        fig.update_yaxes(title_text=yaxis_titles[i], row=i+2, col=1)

    # X轴：每月刻度 + 年份倾斜45° + 最新高亮
    monthly_ticks = pd.date_range(f"{start_year}-01-01", datetime.now(), freq='MS')
    latest = dates[-1]
    if latest not in monthly_ticks:
        monthly_ticks = monthly_ticks.union([latest])

    tick_text = []
    for d in monthly_ticks:
        if d == latest:
            tick_text.append(f"最新\n{d.strftime('%Y-%m-%d')}")
        elif d.month == 1:
            tick_text.append(d.strftime('%Y'))
        else:
            tick_text.append(d.strftime('%m'))

    for i in range(1, 8):
        fig.update_xaxes(
            tickmode='array',
            tickvals=monthly_ticks,
            ticktext=tick_text,
            tickangle=45,
            tickfont=dict(size=11),
            range=[dates[0], dates[-1]],
            row=i, col=1
        )

    fig.update_layout(height=2800, title=f"{'2025 当年' if start_year == 2025 else '2015-至今历史'} 实时监控")
    return fig

# ==================== Tab ====================
tab1, tab2 = st.tabs(["2025 当年", "2015-至今历史"])

with tab1:
    st.plotly_chart(make_chart(2025), use_container_width=True)

with tab2:
    st.plotly_chart(make_chart(2015), use_container_width=True)

st.caption(f"数据更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M')} | BTC 曲线强制显示，永不消失！")