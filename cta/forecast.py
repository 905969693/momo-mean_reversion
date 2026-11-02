"""方向：时间序列动量，以及均线 + ADX 的强趋势对照。"""
import numpy as np
import pandas as pd

from cta.config import ADX_N, ADX_TH, LOOKBACKS, MA_N


def tsmom_forecast(close, lookbacks=LOOKBACKS):
    """过去 L 日收益的符号。各档都有值才给出 forecast，范围 [-1, 1]。"""
    signs = []
    for lookback in lookbacks:
        lagged = close.shift(lookback)
        returns = close / lagged - 1.0
        signs.append(np.sign(returns))
    frame = pd.concat(signs, axis=1)
    forecast = frame.mean(axis=1)
    return forecast.where(frame.notna().all(axis=1))


def adx_parts(df, n=ADX_N):
    """+DI、-DI、ADX。平滑方式与背离项目里的趋势分区相同。"""
    high, low, close = df.High, df.Low, df.Close
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.clip(lower=0).where(up_move > down_move, 0.0)
    minus_dm = down_move.clip(lower=0).where(down_move > up_move, 0.0)
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / n, adjust=False).mean()
    return plus_di, minus_di, adx_line


def regime_forecast(df, ma_n=MA_N, adx_n=ADX_N, adx_th=ADX_TH):
    """强趋势多 = +1，强趋势空 = -1，震荡 = 0。均线尚未形成时为缺失。"""
    close = df.Close
    ma = close.rolling(ma_n).mean()
    above = close > ma
    plus_di, minus_di, adx_line = adx_parts(df, adx_n)
    strong = adx_line > adx_th
    bull = above & strong & (plus_di > minus_di) & ma.notna()
    bear = (~above) & strong & (minus_di > plus_di) & ma.notna()
    ready = ma.notna() & adx_line.notna() & plus_di.notna() & minus_di.notna()
    forecast = pd.Series(np.nan, index=df.index, dtype=float)
    forecast = forecast.mask(ready, 0.0)
    forecast = forecast.mask(bull, 1.0)
    forecast = forecast.mask(bear, -1.0)
    return forecast
