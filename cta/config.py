"""写死的规则。这一版不做参数搜索。"""
from pathlib import Path

# 只读。不在这里下载，也不把结果写回这份缓存。
CACHE_DIR = Path("/Users/yanyunfeng/文稿/tradingview/cache")
PERIOD_CANDIDATES = ("2y", "5y", "10y", "max")

TICKERS = [
    "SPY",
    "EEM",
    "KBE",
    "TLT",
    "GLD",
    "USDJPY=X",
    "USO",
]

# 约 3 个月、1 年。两档符号取平均。去掉 21 日是为了少换手。
LOOKBACKS = (63, 252)
VOL_SPAN = 32
TARGET_VOL = 0.15
WEIGHT_CAP = 2.0
ANNUALIZATION = 252

# 与现有趋势分区一致：均线 200，ADX 14，强趋势阈值 25。
MA_N = 200
ADX_N = 14
ADX_TH = 25.0

COST_BPS = (0.0, 5.0)
WEEKLY_RULE = "W-FRI"

FORECAST_TSMOM = "时间序列动量"
FORECAST_REGIME = "均线ADX"
FORECASTS = (FORECAST_TSMOM, FORECAST_REGIME)
