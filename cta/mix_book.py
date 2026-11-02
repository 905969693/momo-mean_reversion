"""当前组合的仓位。股票只做底背离，商品顶底都做，外汇只做动量。

每个品种一份等权资金，品种仓位在 [-1, 1]。组合权重是这份仓位除以当天有信号的品种数，
所以总敞口不超过 100%，没有杠杆。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cta.config import CACHE_DIR  # noqa: E402
from cta.backtest import combine_equal_risk, daily_pnl  # noqa: E402
from cta.config import ANNUALIZATION  # noqa: E402
from cta.divergence_book import HOLD_DAYS, KBE_PRESET, _mode, divergence_side, switch_position  # noqa: E402
from cta.forecast import tsmom_forecast  # noqa: E402
from run_fx import NAMES as FX_NAMES, default_preset, ensure_daily  # noqa: E402

GROUPS = {
    "股票": ["SPY", "SOXX"],
    "商品": ["GLD", "SLV", "USO", "UNG", "DBA", "GSG"],
    "外汇": [ticker for ticker in FX_NAMES if ticker.startswith("USD")],
}
GROUP_ORDER = {"股票": 0, "商品": 1, "外汇": 2}
SHORT_OK = {"商品"}


def default_members():
    members = []
    for group, tickers in GROUPS.items():
        for ticker in tickers:
            members.append((ticker, group))
    return members


def _saved_preset(ticker):
    if ticker == "KBE":
        preset = dict(KBE_PRESET)
        preset["ticker"] = ticker
        return preset
    path = CACHE_DIR / "fixed_presets.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    spec = saved.get(ticker)
    if not isinstance(spec, dict):
        return None
    try:
        return {
            "ticker": ticker,
            "pivot_order": int(spec["pivot_order"]),
            "min_count": int(spec["min_count"]),
            "dedup_bars": int(spec["dedup_bars"]),
            "ma_n": int(spec["ma_n"]),
            "adx_th": None if spec.get("adx_th") in (None, "") else int(spec["adx_th"]),
            "mode": _mode(spec),
            "long_only": bool(spec.get("long_only", False)),
            "hold": HOLD_DAYS,
            "summary": str(spec.get("summary") or ""),
        }
    except (KeyError, TypeError, ValueError):
        return None


def presets_for(members=None):
    if members is None:
        members = default_members()
    presets = {}
    groups = {}
    for ticker, group in members:
        if group not in GROUP_ORDER:
            raise ValueError(f"{ticker} 的分组必须是股票、商品或外汇")
        preset = _saved_preset(ticker) or default_preset(ticker)
        preset = dict(preset)
        preset["ticker"] = ticker
        if group == "外汇":
            preset["summary"] = "只做动量，不做背离"
        else:
            preset["long_only"] = group not in SHORT_OK
            text = preset["summary"].replace("、只做多", "").replace("、多空都做", "")
            preset["summary"] = text + ("、多空都做" if group in SHORT_OK else "、只做底背离")
        presets[ticker] = preset
        groups[ticker] = group
    return presets, groups


def _source_frame(weight, side):
    source = pd.Series(pd.NA, index=weight.index, dtype="object")
    live = weight.notna()
    source = source.mask(live, "动量")
    source = source.mask(live & (side > 0), "底背离")
    source = source.mask(live & (side < 0), "顶背离")
    return source


def compute_sleeves(markets, presets, groups):
    weights = {}
    sources = {}
    closes = {}
    for ticker, daily in markets.items():
        forecast = tsmom_forecast(daily.Close).clip(-1, 1)
        if groups[ticker] == "外汇":
            weight = forecast
            source = pd.Series("动量", index=daily.index, dtype="object").where(weight.notna())
        else:
            weight = switch_position(daily, presets[ticker], forecast)
            side, _events = divergence_side(daily, presets[ticker], both_sides=False)
            source = _source_frame(weight, side.reindex(weight.index).fillna(0.0))
        weights[ticker] = weight
        sources[ticker] = source
        closes[ticker] = daily.Close
    weight_frame = pd.DataFrame(weights).sort_index()
    source_frame = pd.DataFrame(sources).sort_index()
    close_frame = pd.DataFrame(closes).sort_index()
    finite = weight_frame.abs().to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) and float(finite.max()) > 1.0 + 1e-9:
        raise RuntimeError("品种仓位超过 1，这套组合不加杠杆")
    return weight_frame, source_frame, close_frame


def portfolio_weights(weight_frame):
    """当天有信号的品种等权。空仓是 0，仍占一份；还在预热的缺失不占。"""
    count = weight_frame.notna().sum(axis=1)
    return weight_frame.div(count.where(count > 0), axis=0)


def book_date_of(weight_frame):
    """多数品种都有行情的最近一天。个别品种多出来的报价不单独把组合往后推。"""
    last = []
    for column in weight_frame.columns:
        valid = weight_frame[column].dropna()
        if not valid.empty:
            last.append(valid.index[-1])
    if not last:
        raise RuntimeError("没有可用的仓位")
    modes = pd.Series(last).mode()
    return pd.Timestamp(modes.max())


def _value_on(series, day, grace_days=5):
    if day in series.index and pd.notna(series.loc[day]):
        value = series.loc[day]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return value, pd.Timestamp(day)
    prior = series.loc[:day].dropna()
    if prior.empty:
        return np.nan, pd.NaT
    stamp = pd.Timestamp(prior.index[-1])
    if (pd.Timestamp(day) - stamp).days <= grace_days:
        return prior.iloc[-1], stamp
    return np.nan, stamp


def _direction(weight):
    if pd.isna(weight):
        return pd.NA
    if weight > 1e-9:
        return "多"
    if weight < -1e-9:
        return "空"
    return "空仓"


def _sign(weight):
    if weight > 1e-9:
        return 1
    if weight < -1e-9:
        return -1
    return 0


def _action(old, new):
    if old == 0 and new > 0:
        return "开多"
    if old == 0 and new < 0:
        return "开空"
    if old > 0 and new == 0:
        return "平多"
    if old < 0 and new == 0:
        return "平空"
    if old > 0 and new < 0:
        return "翻空"
    if old < 0 and new > 0:
        return "翻多"
    return "调整"


def snapshot_table(weight_frame, source_frame, close_frame, presets, groups, day):
    port = portfolio_weights(weight_frame)
    rows = []
    for ticker in weight_frame.columns:
        weight, weight_day = _value_on(weight_frame[ticker], day)
        if pd.isna(weight):
            continue
        source, _source_day = _value_on(source_frame[ticker], day)
        close, _close_day = _value_on(close_frame[ticker], day)
        book_weight, _book_day = _value_on(port[ticker], weight_day)
        rows.append({
            "分组": groups[ticker],
            "标的": ticker,
            "方向": _direction(weight),
            "品种仓位": float(weight),
            "组合权重": float(book_weight),
            "来源": source,
            "规则": presets[ticker]["summary"],
            "收盘": float(close) if pd.notna(close) else np.nan,
            "行情日": pd.Timestamp(weight_day).date(),
            "_顺序": GROUP_ORDER[groups[ticker]],
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["_顺序", "标的"]).drop(columns="_顺序").reset_index(drop=True)


def change_table(weight_frame, source_frame, groups, day, sessions=30):
    """最近 sessions 个有行情的交易日里，品种仓位发生变化的记录。"""
    count = weight_frame.notna().sum(axis=1)
    active = weight_frame.index[weight_frame.notna().any(axis=1)]
    active = active[active <= day]
    window = active[-sessions:]
    rows = []
    for ticker in weight_frame.columns:
        held = weight_frame[ticker].dropna()
        if held.empty:
            continue
        previous = held.shift(1)
        previous_day = held.index.to_series().shift(1)
        changed = previous.notna() & ((held - previous).abs() > 1e-9) & held.index.isin(window)
        for stamp in held.index[changed]:
            old = float(previous.loc[stamp])
            new = float(held.loc[stamp])
            prior = pd.Timestamp(previous_day.loc[stamp])
            rows.append({
                "日期": pd.Timestamp(stamp).date(),
                "分组": groups[ticker],
                "标的": ticker,
                "动作": _action(_sign(old), _sign(new)),
                "原仓位": old,
                "新仓位": new,
                "原组合权重": old / float(count.loc[prior]),
                "新组合权重": new / float(count.loc[stamp]),
                "原来源": source_frame[ticker].loc[prior],
                "新来源": source_frame[ticker].loc[stamp],
                "_顺序": pd.Timestamp(stamp),
                "_组": GROUP_ORDER[groups[ticker]],
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return (
        table.sort_values(["_顺序", "_组", "标的"], ascending=[False, True, True])
        .drop(columns=["_顺序", "_组"])
        .reset_index(drop=True)
    )


def portfolio_daily_pnl(weight_frame, markets, cost_bps=0.0):
    """切换策略下每个品种等权合成的日盈亏。"""
    pnls = {}
    for ticker, daily in markets.items():
        if ticker not in weight_frame.columns:
            continue
        weight = weight_frame[ticker]
        asset_return = daily.Close.pct_change().reindex(weight.index)
        pnls[ticker] = daily_pnl(weight, asset_return, cost_bps)
    return combine_equal_risk(pnls)


def _spy_nav_on(index):
    try:
        daily, _meta = ensure_daily("SPY")
    except Exception:
        return pd.Series(np.nan, index=index, dtype=float)
    close = daily.Close.reindex(index)
    if close.isna().all():
        return pd.Series(np.nan, index=index, dtype=float)
    base = close.dropna().iloc[0]
    return close / base


def pnl_returns_table(weight_frame, markets, cost_bps=0.0):
    """有组合盈亏的全部交易日，只存日收益，便于前端切区间。"""
    book = portfolio_daily_pnl(weight_frame, markets, cost_bps=cost_bps)
    realized = book.dropna()
    if realized.empty:
        return pd.DataFrame(columns=["日期", "日盈亏"])
    return pd.DataFrame({
        "日期": pd.DatetimeIndex(realized.index),
        "日盈亏": realized.to_numpy(),
    })


def slice_pnl_history(returns_table, trading_days=None):
    """截取最近 trading_days 根（None 表示全部），组合与 SPY 都在区间起点归一化为 1。"""
    if returns_table is None or returns_table.empty:
        return pd.DataFrame(columns=["日期", "日盈亏", "净值", "SPY"])
    view = returns_table if trading_days is None else returns_table.tail(int(trading_days))
    if view.empty:
        return pd.DataFrame(columns=["日期", "日盈亏", "净值", "SPY"])
    daily = view["日盈亏"]
    nav = (1.0 + daily).cumprod()
    nav = nav / nav.iloc[0]
    dates = pd.DatetimeIndex(view["日期"])
    spy = _spy_nav_on(dates)
    return pd.DataFrame({
        "日期": dates,
        "日盈亏": daily.to_numpy(),
        "净值": nav.to_numpy(),
        "SPY": spy.to_numpy(),
    })


def pnl_history(weight_frame, markets, trading_days=ANNUALIZATION, cost_bps=0.0):
    table = pnl_returns_table(weight_frame, markets, cost_bps=cost_bps)
    return slice_pnl_history(table, trading_days=trading_days)


def exposure_table(weight_frame, day, sessions=30):
    port = portfolio_weights(weight_frame)
    active = port.index[weight_frame.notna().any(axis=1)]
    active = active[active <= day]
    window = active[-sessions:]
    view = port.loc[window]
    return pd.DataFrame({
        "日期": pd.DatetimeIndex(window),
        "净敞口": view.sum(axis=1, min_count=1).to_numpy(),
        "多头敞口": view.clip(lower=0).sum(axis=1, min_count=1).to_numpy(),
        "空头敞口": view.clip(upper=0).sum(axis=1, min_count=1).to_numpy(),
    })


def _empty_book(failed, names):
    return {
        "snapshot": pd.DataFrame(),
        "changes": pd.DataFrame(),
        "exposure": pd.DataFrame(columns=["日期", "净敞口", "多头敞口", "空头敞口"]),
        "pnl_returns": pd.DataFrame(columns=["日期", "日盈亏"]),
        "book_date": None,
        "window_start": None,
        "window_end": None,
        "latest_quote": None,
        "names": names,
        "failed": failed,
    }


def build_book(members=None, sessions=30):
    if members is None:
        members = default_members()
    members = list(dict.fromkeys((ticker, group) for ticker, group in members))
    presets, groups = presets_for(members)
    markets = {}
    failed = []
    for ticker in presets:
        try:
            daily, _meta = ensure_daily(ticker)
        except Exception as exc:
            failed.append(f"{ticker}：{exc}")
            continue
        if daily.empty:
            failed.append(f"{ticker}：没有日线")
            continue
        markets[ticker] = daily
    if not markets:
        return _empty_book(failed, len(presets))
    live_presets = {ticker: presets[ticker] for ticker in markets}
    live_groups = {ticker: groups[ticker] for ticker in markets}
    weights, sources, closes = compute_sleeves(markets, live_presets, live_groups)
    try:
        day = book_date_of(weights)
    except RuntimeError:
        return _empty_book(failed, len(presets))
    latest = max(series.dropna().index[-1] for series in (weights[column] for column in weights.columns))
    snap = snapshot_table(weights, sources, closes, live_presets, live_groups, day)
    changes = change_table(weights, sources, live_groups, day, sessions=sessions)
    exposure = exposure_table(weights, day, sessions=sessions)
    pnl_returns = pnl_returns_table(weights, markets)
    gross = float(snap["组合权重"].abs().sum()) if not snap.empty else 0.0
    if gross > 1.0 + 1e-6:
        raise RuntimeError(f"组合总敞口 {gross:.2%} 超过 100%，需要杠杆")
    window = exposure["日期"]
    return {
        "snapshot": snap,
        "changes": changes,
        "exposure": exposure,
        "pnl_returns": pnl_returns,
        "book_date": pd.Timestamp(day).date().isoformat(),
        "window_start": pd.Timestamp(window.iloc[0]).date().isoformat() if len(window) else None,
        "window_end": pd.Timestamp(window.iloc[-1]).date().isoformat() if len(window) else None,
        "latest_quote": pd.Timestamp(latest).date().isoformat(),
        "names": len(presets),
        "failed": failed,
    }
