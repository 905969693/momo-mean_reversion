"""只读现有日线缓存，并按周五收盘重采样周线。"""
import re

import pandas as pd

from cta.config import CACHE_DIR, PERIOD_CANDIDATES, WEEKLY_RULE

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def safe_chart_name(ticker):
    return re.sub(r"[^\w.\-]+", "_", ticker)


def read_ohlcv(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "")
        if key in ("open", "high", "low", "close", "volume"):
            rename[col] = "Close" if key == "close" else key.capitalize()
    df = df.rename(columns=rename)
    missing = [col for col in OHLCV if col not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列 {missing}，实际列为 {list(df.columns)}")
    out = df.loc[:, OHLCV].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise ValueError(f"{path} 没有可用的日线")
    return out


def select_daily(ticker, cache_dir=CACHE_DIR):
    """在 2y/5y/10y/max 中选用起始日最早的一份；起始日相同则留行数更多的。"""
    name = safe_chart_name(ticker)
    best = None
    for period in PERIOD_CANDIDATES:
        path = cache_dir / f"{name}_{period}.csv"
        if not path.is_file():
            continue
        df = read_ohlcv(path)
        rank = (df.index[0], -len(df))
        if best is None or rank < best[0]:
            best = (rank, period, path, df)
    if best is None:
        raise FileNotFoundError(
            f"{ticker} 在 {cache_dir} 没有日线缓存"
            f"（查找 {name}_2y/5y/10y/max.csv）。请先在现有 Dashboard 里补缓存。"
        )
    _, period, path, df = best
    return df, period, path


def to_weekly(daily, rule=WEEKLY_RULE, drop_partial=True):
    """日线聚合成周线。最后一周若还没到周五，丢掉。"""
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    weekly = daily.resample(rule).agg(agg).dropna()
    if (
        drop_partial
        and len(weekly)
        and len(daily)
        and pd.Timestamp(weekly.index[-1]).normalize()
        > pd.Timestamp(daily.index[-1]).normalize()
    ):
        weekly = weekly.iloc[:-1]
    return weekly


def load_market(ticker, cache_dir=CACHE_DIR):
    daily, period, path = select_daily(ticker, cache_dir)
    weekly = to_weekly(daily)
    if len(weekly) and pd.Timestamp(weekly.index[-1]).normalize() > pd.Timestamp(daily.index[-1]).normalize():
        raise RuntimeError(f"{ticker} 周线最后一根晚于日线")
    meta = {
        "标的": ticker,
        "区间": period,
        "缓存文件": str(path),
        "日线起始": daily.index[0].date().isoformat(),
        "日线结束": daily.index[-1].date().isoformat(),
        "日线根数": len(daily),
        "周线起始": weekly.index[0].date().isoformat() if len(weekly) else "",
        "周线结束": weekly.index[-1].date().isoformat() if len(weekly) else "",
        "周线根数": len(weekly),
    }
    return daily, weekly, meta
