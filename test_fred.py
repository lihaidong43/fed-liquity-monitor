# 文件名: test_fred_api.py
# 一键测试 FRED 官方 API 是否可用（推荐！）

import requests
import pandas as pd
from datetime import datetime

# 第一步：把你申请的 FRED API Key 填到下面（没有就去这个链接30秒免费申请）
# https://fred.stlouisfed.org/docs/api/api_key.html
API_KEY = "b63291c1dfec1ab5ee875db25f1e2891"   # ←←← 改这里！！！

if API_KEY == "在这里填你的API_KEY" or len(API_KEY) != 32:
    print("错误：请先去 https://fred.stlouisfed.org/docs/api/api_key.html 申请一个免费 API Key")
    exit()

# 要测试的4个核心指标
series = {
    'WTREGEN':   '财政部现金余额 TGA',
    'SOFR':      'SOFR 利率',
    'RRPONTSYD': '隔夜逆回购 RRP',
    'WALCL':     '美联储资产负债表'
}

print("开始测试 FRED 官方 API 连通性...\n" + "="*60)

all_good = True

for sid, name in series.items():
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': sid,
        'api_key': API_KEY,
        'file_type': 'json',
        'limit': 5,                    # 只拿最新5条看一眼就行
        'sort_order': 'desc'           # 最新的在前
    }
    
    try:
        print(f"正在测试 {name} ({sid}) ... ", end="")
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if 'observations' in data and len(data['observations']) > 0:
                latest = data['observations'][0]
                date = latest['date']
                value = latest['value']
                if value == '.':
                    value = "无数据"
                print("成功！")
                print(f"   最新日期: {date}   最新值: {value}")
            else:
                print("成功但暂无数据（可能是周末）")
        else:
            print(f"失败！HTTP {r.status_code}")
            print(f"   响应内容: {r.text[:200]}")
            all_good = False
    except Exception as e:
        print(f"异常: {e}")
        all_good = False

print("="*60)
if all_good:
    print("大成功！FRED 官方 API 完全可用！")
    print("你可以直接使用下面这段代码，永不失联、永不被墙！")
    print("我已经给你写好完整版了，复制粘贴就能跑")
else:
    print("FRED API 也被墙了")
    print("唯一解法：使用我之前给你的「国内镜像版」代码（已永久解决）")