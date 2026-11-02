"""滞后一根的逐日盈亏、汇总指标、净值图。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from cta.config import ANNUALIZATION

_CJK = (
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
    "SimHei",
    "Microsoft YaHei",
)
_available = {font.name for font in font_manager.fontManager.ttflist}
for _name in _CJK:
    if _name in _available:
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False


def daily_pnl(weight, asset_return, cost_bps):
    """第 t 日收盘的权重赚 t 到 t+1 的收益。成本按换手的单边 bps。"""
    held = weight.shift(1)
    previous = weight.shift(2)
    previous_filled = previous.where(previous.notna(), 0.0)
    turnover = (held - previous_filled).abs()
    cost = turnover * (cost_bps / 1e4)
    return held * asset_return - cost


def assert_lagged(weight, asset_return, pnl):
    """零成本时，盈亏必须等于昨日权重乘今日收益。"""
    expected = weight.shift(1) * asset_return
    both = pd.concat([expected.rename("expected"), pnl.rename("pnl")], axis=1).dropna()
    if both.empty:
        raise AssertionError("没有可核对的盈亏")
    if not np.allclose(both["expected"], both["pnl"], equal_nan=True):
        raise AssertionError("盈亏没有滞后一根")


def wealth(pnl):
    realized = pnl.dropna()
    if realized.empty:
        return realized
    return (1.0 + realized).cumprod()


def annualized_return(pnl, annualization=ANNUALIZATION):
    realized = pnl.dropna()
    if realized.empty:
        return np.nan
    growth = (1.0 + realized).prod()
    if growth <= 0:
        return np.nan
    years = len(realized) / annualization
    if years <= 0:
        return np.nan
    return float(growth ** (1.0 / years) - 1.0)


def annualized_vol(pnl, annualization=ANNUALIZATION):
    realized = pnl.dropna()
    if len(realized) < 2:
        return np.nan
    return float(realized.std() * np.sqrt(annualization))


def sharpe(pnl, annualization=ANNUALIZATION):
    realized = pnl.dropna()
    if len(realized) < 2:
        return np.nan
    scale = realized.std()
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(realized.mean() / scale * np.sqrt(annualization))


def max_drawdown(pnl):
    curve = wealth(pnl)
    if curve.empty:
        return np.nan
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def position_mix(weight):
    active = weight.dropna()
    if active.empty:
        return np.nan, np.nan, np.nan
    return (
        float((active > 0).mean()),
        float((active < 0).mean()),
        float((active == 0).mean()),
    )


def average_holding_days(weight):
    """同方向连续持仓的平均天数。空仓不计入。"""
    active = weight.dropna()
    if active.empty:
        return np.nan
    signs = np.sign(active.to_numpy())
    lengths = []
    start = 0
    count = len(signs)
    while start < count:
        if signs[start] == 0:
            start += 1
            continue
        end = start + 1
        while end < count and signs[end] == signs[start]:
            end += 1
        lengths.append(end - start)
        start = end
    if not lengths:
        return np.nan
    return float(np.mean(lengths))


def performance_row(label, forecast_name, cost_bps, pnl, weight=None):
    realized = pnl.dropna()
    long_pct = short_pct = flat_pct = hold = np.nan
    if weight is not None:
        long_pct, short_pct, flat_pct = position_mix(weight)
        hold = average_holding_days(weight)
    return {
        "标的": label,
        "预测": forecast_name,
        "成本bps": cost_bps,
        "年化收益": annualized_return(pnl),
        "年化波动": annualized_vol(pnl),
        "夏普": sharpe(pnl),
        "最大回撤": max_drawdown(pnl),
        "多头占比": long_pct,
        "空头占比": short_pct,
        "空仓占比": flat_pct,
        "平均持仓天数": hold,
        "样本天数": int(len(realized)),
        "起始": realized.index[0].date().isoformat() if len(realized) else "",
        "结束": realized.index[-1].date().isoformat() if len(realized) else "",
    }


def combine_equal_risk(pnl_by_ticker):
    """每个品种已经波动率缩放。当天有盈亏的品种等权平均。"""
    frame = pd.DataFrame(pnl_by_ticker).sort_index()
    count = frame.notna().sum(axis=1)
    total = frame.sum(axis=1, min_count=1)
    return total / count.where(count > 0)


def combine_by_risk(pnl_by_book, window=756, min_periods=252):
    """两本账按截至昨天的滚动波动分配资金，波动低的权重大。

    权重之和为 1。某本账的波动还没形成时，当天的资金全部给已经有波动的账。
    返回组合盈亏，以及各账权重。
    """
    frame = pd.DataFrame(pnl_by_book).sort_index()
    vol = frame.rolling(window, min_periods=min_periods).std().shift(1)
    inverse = 1.0 / vol.where(vol > 0)
    inverse = inverse.where(frame.notna())
    weights = inverse.div(inverse.sum(axis=1), axis=0)
    combined = (weights * frame).sum(axis=1, min_count=1)
    return combined, weights


def scale_to_target(pnl, target, window=756, min_periods=252, cap=8.0):
    """用截至昨天的滚动波动，把一条盈亏放大到目标年化波动。"""
    trailing = pnl.rolling(window, min_periods=min_periods).std().shift(1)
    annual = trailing * np.sqrt(ANNUALIZATION)
    scale = (target / annual.where(annual > 0)).clip(upper=cap)
    return pnl * scale


def _style_axis(ax, title):
    ax.set_title(title)
    ax.set_ylabel("净值")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.figure.autofmt_xdate()
    ax.figure.tight_layout()


def plot_equity(path, title, series):
    """series: (名字, 盈亏序列, 线型) 。每条线从自己的第一天重定为 1。"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, pnl, style in series:
        curve = wealth(pnl)
        if curve.empty:
            continue
        ax.plot(curve.index, curve.to_numpy(), style, lw=1.4, label=name)
    _style_axis(ax, title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def format_summary(frame):
    shown = frame.copy()
    percent_cols = ["年化收益", "年化波动", "最大回撤", "多头占比", "空头占比", "空仓占比"]
    for col in percent_cols:
        shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{value * 100:.2f}%")
    shown["夏普"] = shown["夏普"].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    shown["平均持仓天数"] = shown["平均持仓天数"].map(
        lambda value: "" if pd.isna(value) else f"{value:.1f}"
    )
    return shown
