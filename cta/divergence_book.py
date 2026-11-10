"""各品种的背离账。KBE 用十年回测固定参数，其余读 Dashboard 已保存的参数。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cta.config import CACHE_DIR, TARGET_VOL, TICKERS
from cta.position import annualized_vol

HOLD_DAYS = 10  # 某一侧没有尖峰时才用。正常持有期是该侧超额平均最高的根数。

# 逆势、ADX 关、MA100、pivot 6、共振 4、去重 10。多空都做。
KBE_PRESET = {
    "ticker": "KBE",
    "pivot_order": 6,
    "min_count": 4,
    "dedup_bars": 10,
    "ma_n": 100,
    "adx_th": None,
    "mode": "counter",
    "long_only": False,
    "hold": HOLD_DAYS,
    "summary": "逆势、ADX 关、MA100、pivot 6、共振 4、去重 10",
}


def _tradingview():
    """本地开发可读 tradingview 项目；Streamlit Cloud 等环境用仓库内副本。"""
    roots = []
    tv_cache_parent = CACHE_DIR.parent
    if tv_cache_parent.is_dir():
        roots.append(tv_cache_parent)
    repo_root = Path(__file__).resolve().parents[1]
    roots.append(repo_root)
    for root in roots:
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)
        try:
            import tradingview_realization_v2 as tv
            return tv
        except ImportError:
            continue
    raise ImportError(
        "找不到 tradingview_realization_v2（背离信号）。"
        "请确认仓库根目录有 tradingview_realization_v2.py。"
    )


def _mode(spec):
    if not spec["trend_on"]:
        return "off"
    if not spec["aligned_keep"]:
        return "counter"
    return "aligned"


def load_presets(tickers=TICKERS):
    saved = json.loads((CACHE_DIR / "fixed_presets.json").read_text())
    presets = {}
    for ticker in tickers:
        if ticker == "KBE":
            presets[ticker] = dict(KBE_PRESET)
            continue
        spec = saved[ticker]
        presets[ticker] = {
            "ticker": ticker,
            "pivot_order": int(spec["pivot_order"]),
            "min_count": int(spec["min_count"]),
            "dedup_bars": int(spec["dedup_bars"]),
            "ma_n": int(spec["ma_n"]),
            "adx_th": None if spec["adx_th"] in (None, "") else int(spec["adx_th"]),
            "mode": _mode(spec),
            "long_only": bool(spec.get("long_only", False)),
            "hold": HOLD_DAYS,
            "summary": spec["summary"],
        }
    return presets


def divergence_sign(daily, preset):
    """有多头仓位为 +1，有空头仓位为 -1。信号确认后的下一根才开始计收益。"""
    tv = _tradingview()
    bull, bear = tv.raw_deduped_signals(
        daily,
        pivot_order=preset["pivot_order"],
        min_count=preset["min_count"],
        dedup_bars=preset["dedup_bars"],
    )
    bull, bear = tv.apply_trend_filter(
        daily, bull, bear,
        ma_n=preset["ma_n"],
        mode=preset["mode"],
        adx_th=preset["adx_th"],
    )
    if preset["long_only"]:
        bear = bear.iloc[0:0]
    n = len(daily)
    long_on = np.zeros(n)
    short_on = np.zeros(n)
    order = preset["pivot_order"]
    bull_hold, bear_hold = peak_holds(daily, preset)

    def occupy(signals, active, hold):
        busy_until = -1
        for dt in signals.index:
            loc = daily.index.get_loc(dt)
            if isinstance(loc, slice):
                loc = loc.start
            entry = int(loc) + order
            if entry >= n or entry <= busy_until:
                continue
            start = entry + 1
            end = min(entry + hold, n - 1)
            if start <= end:
                active[start:end + 1] = 1.0
            busy_until = entry + hold

    occupy(bull, long_on, bull_hold)
    occupy(bear, short_on, bear_hold)
    return pd.Series(long_on - short_on, index=daily.index)


def divergence_weight(daily, preset, target=TARGET_VOL):
    """有仓时按目标波动缩放，空仓为 0。"""
    sign = divergence_sign(daily, preset)
    vol = annualized_vol(daily.Close)
    weight = (sign * (target / vol.where(vol > 0))).clip(-2, 2)
    return weight.where(vol.notna())


def _bar_loc(index, dt):
    loc = index.get_loc(dt)
    if isinstance(loc, slice):
        return int(loc.start)
    return int(loc)


def filtered_signals(daily, preset):
    tv = _tradingview()
    bull, bear = tv.raw_deduped_signals(
        daily,
        pivot_order=preset["pivot_order"],
        min_count=preset["min_count"],
        dedup_bars=preset["dedup_bars"],
    )
    bull, bear = tv.apply_trend_filter(
        daily, bull, bear,
        ma_n=preset["ma_n"],
        mode=preset["mode"],
        adx_th=preset["adx_th"],
    )
    if preset["long_only"]:
        bear = bear.iloc[0:0]
    return bull, bear


_PEAK_SPAN = tuple(range(3, 41))


def _mean_excess(tv, daily, sig, horizon, entry_shift, short):
    rets = tv.signal_returns(
        daily, sig, horizon, entry_shift=entry_shift, short=short,
    )
    base = tv.unconditional_returns(daily, horizon, short=short)
    if getattr(rets, "size", 0) == 0 or getattr(base, "size", 0) == 0:
        return None
    if not np.isfinite(rets).any() or not np.isfinite(base).all():
        return None
    mean_sig = float(np.mean(rets)) if np.isfinite(rets).all() else float(np.nanmean(rets))
    base_mean = float(np.mean(base))
    if not np.isfinite(mean_sig) or not np.isfinite(base_mean):
        return None
    return mean_sig - base_mean


def _peak_horizon(tv, daily, sig, entry_shift, short):
    """超额平均最高的持有根数。并列时取更短的那根。"""
    best_h = None
    best_v = None
    for horizon in _PEAK_SPAN:
        excess = _mean_excess(tv, daily, sig, horizon, entry_shift, short)
        if excess is None:
            continue
        if best_v is None or excess > best_v:
            best_h = horizon
            best_v = excess
    return HOLD_DAYS if best_h is None else int(best_h)


def peak_holds(daily, preset, both_sides=False):
    """底、顶各自取超额平均最高的持有根数。没有样本时退回 10 根。

    尖峰在这整段日线上选，和后面的盈亏用的是同一段历史。
    只调用 signal_returns 和 unconditional_returns，不依赖后加的曲线函数。
    """
    tv = _tradingview()
    rules = dict(preset)
    if both_sides:
        rules["long_only"] = False
    bull, bear = filtered_signals(daily, rules)
    shift = int(rules["pivot_order"])
    bull_peak = _peak_horizon(tv, daily, bull, shift, short=False)
    if rules["long_only"]:
        bear_peak = HOLD_DAYS
    else:
        bear_peak = _peak_horizon(tv, daily, bear, shift, short=True)
    return int(bull_peak), int(bear_peak)


def _divergence_events(daily, preset, both_sides=False):
    rules = dict(preset)
    if both_sides:
        rules["long_only"] = False
    bull, bear = filtered_signals(daily, rules)
    bull_hold, bear_hold = peak_holds(daily, rules)
    n = len(daily)
    order = rules["pivot_order"]
    events = []
    for timestamp in bull.index:
        confirm = _bar_loc(daily.index, timestamp) + order
        if confirm < n:
            events.append((confirm, 1.0, bull_hold))
    for timestamp in bear.index:
        confirm = _bar_loc(daily.index, timestamp) + order
        if confirm < n:
            events.append((confirm, -1.0, bear_hold))
    events.sort()
    return events, n


def _apply_divergence_holds(position, events, n):
    for confirm, side, hold in events:
        position[confirm:min(confirm + int(hold), n)] = side
    return position


def divergence_side(daily, preset, both_sides=False):
    """确认日收盘起的方向：底背离 +1，顶背离 -1。持有该侧的尖峰根数。后出现的信号覆盖先前的。"""
    events, n = _divergence_events(daily, preset, both_sides=both_sides)
    side = _apply_divergence_holds(np.zeros(n), events, n)
    return pd.Series(side, index=daily.index), events


def divergence_only_position(daily, preset, both_sides=False):
    """只在背离持有期有仓，其余时间为 0。both_sides 为假时只保留底背离。"""
    events, n = _divergence_events(daily, preset, both_sides=both_sides)
    position = _apply_divergence_holds(np.zeros(n), events, n)
    return pd.Series(position, index=daily.index)


def switch_position(daily, preset, momentum):
    """没有背离时用动量。背离确认日收盘起改为背离方向，持有该侧尖峰根数后再回到动量。

    动量和背离方向的仓位都在 [-1, 1]。1 表示这份品种资金全部做多，不含杠杆。
    股票只保留底背离。商品底和顶各自用自己的尖峰根数。
    """
    events, n = _divergence_events(daily, preset, both_sides=False)
    position = momentum.reindex(daily.index).to_numpy(dtype=float).copy()
    _apply_divergence_holds(position, events, n)
    out = pd.Series(position, index=daily.index)
    finite = out.abs().dropna()
    if len(finite) and float(finite.max()) > 1.0 + 1e-9:
        raise RuntimeError(f"{preset['ticker']} 仓位超过 1，需要杠杆")
    return out
