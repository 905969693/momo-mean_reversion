"""写死的规则。这一版不做参数搜索。"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LOCAL_TV = Path("/Users/yanyunfeng/文稿/tradingview/cache")
_env = os.environ.get("CTA_CACHE_DIR")
if _env:
    CACHE_DIR = Path(_env)
elif _LOCAL_TV.is_dir():
    # 本地：只读 tradingview 已缓存的日线与 fixed_presets.json
    CACHE_DIR = _LOCAL_TV
else:
    # Streamlit Cloud 等：用本项目 cache/（日线由 yfinance 写入）
    CACHE_DIR = _REPO / "cache"
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
