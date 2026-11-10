"""等权持有。底背离把该品种调到 2/n，顶背离调到 0，差额由其余品种平分。

总权重不超过 1，不做空、不加杠杆。多只底背离叠在一起、目标之和超过 1 时，按比例缩小。
六只同时顶背离时没有接收方，当天持有现金。只在信号状态变化的收盘才调仓，中间让仓位随价格漂移。
信号沿用各品种已保存的共振、去重和趋势过滤，但顶背离也生效。确认日收盘起持有该侧尖峰根数。
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cta.backtest import format_summary, performance_row, plot_equity  # noqa: E402
from cta.config import COST_BPS, TICKERS  # noqa: E402
from cta.data import load_market  # noqa: E402
from cta.divergence_book import divergence_side, load_presets  # noqa: E402

BOOST = 2.0
CUT = 0.0


def target_weights(side, boost=BOOST, cut=CUT):
    """有信号的品种钉在目标权重上，没信号的品种平分剩下的资金。

    若目标权重之和超过 1，按比例缩小，避免杠杆。返回权重和是否发生了缩小。
    """
    names = list(side.index)
    n = len(names)
    base = 1.0 / n
    pinned = {}
    free = []
    for name in names:
        flag = float(side[name])
        if flag > 0:
            pinned[name] = base * boost
        elif flag < 0:
            pinned[name] = base * cut
        else:
            free.append(name)
    pinned_sum = sum(pinned.values())
    scaled = False
    weights = {}
    if not free:
        if pinned_sum <= 1e-12:
            for name in names:
                weights[name] = 0.0
        else:
            scale = 1.0 / pinned_sum
            scaled = abs(scale - 1.0) > 1e-12
            for name in names:
                weights[name] = pinned[name] * scale
    elif pinned_sum > 1.0 + 1e-12:
        scaled = True
        scale = 1.0 / pinned_sum
        for name, value in pinned.items():
            weights[name] = value * scale
        for name in free:
            weights[name] = 0.0
    else:
        share = (1.0 - pinned_sum) / len(free)
        weights.update(pinned)
        for name in free:
            weights[name] = share
    out = pd.Series(weights, index=names, dtype=float)
    total = float(out.sum())
    if total > 1.0 + 1e-9 or float(out.min()) < -1e-12 or float(out.max()) > 1.0 + 1e-9:
        raise RuntimeError("权重超过 1、出现空头，或单品种超过全部资金")
    if abs(total - 1.0) > 1e-9 and abs(total) > 1e-9:
        raise RuntimeError("权重之和既不是 1 也不是 0")
    return out, scaled


def _check_targets():
    names = list("ABCDEF")
    n = len(names)
    side = pd.Series(0.0, index=names)
    side["A"] = 1.0
    weights, scaled = target_weights(side)
    if scaled or abs(weights["A"] - BOOST / n) > 1e-12:
        raise AssertionError("底背离没有把仓位设成 2/n")
    expected = (1.0 - BOOST / n) / (n - 1)
    if abs(weights["B"] - expected) > 1e-12:
        raise AssertionError("底背离的差额没有平均分给其余品种")
    side["A"] = -1.0
    weights, scaled = target_weights(side)
    if scaled or abs(weights["A"] - CUT / n) > 1e-12:
        raise AssertionError("顶背离没有把仓位设成 0")
    expected = (1.0 - CUT / n) / (n - 1)
    if abs(weights["B"] - expected) > 1e-12:
        raise AssertionError("顶背离多出的资金没有平均分给其余品种")


def build_frame(names, presets):
    returns = {}
    sides = {}
    counts = {}
    for ticker in names:
        daily, _weekly, _meta = load_market(ticker)
        returns[ticker] = daily.Close.pct_change()
        side, events = divergence_side(daily, presets[ticker], both_sides=True)
        sides[ticker] = side
        counts[ticker] = (sum(flag > 0 for _at, flag in events), sum(flag < 0 for _at, flag in events))
    full_returns = pd.DataFrame(returns).sort_index()
    full_sides = pd.DataFrame(sides).sort_index().ffill().fillna(0.0)
    start = full_returns.apply(lambda column: column.first_valid_index()).max()
    held_returns = full_returns.loc[start:].fillna(0.0)
    held_sides = full_sides.reindex(held_returns.index).ffill().fillna(0.0)
    if (held_returns <= -1.0).any().any():
        raise RuntimeError("存在 -100% 或更差的日收益，持有权重会失真")
    return held_returns, held_sides, counts


def simulate(returns, sides, cost_bps):
    """第一天收盘按当天信号建仓，不收建仓成本。之后只在信号状态变化时调仓。"""
    wealth = 1.0
    weights = None
    previous = None
    scaled_days = 0
    cash_days = 0
    rebalances = 0
    tilt_days = 0
    overlap_days = 0
    max_weight = 0.0
    rows = []
    for timestamp, side in sides.iterrows():
        target, scaled = target_weights(side)
        scaled_days += int(scaled)
        cash_days += int(float(target.sum()) <= 1e-12)
        active = int((side != 0).sum())
        tilt_days += int(active > 0)
        overlap_days += int(active > 1)
        state = side.to_numpy()
        if weights is None:
            weights = target.to_numpy(dtype=float)
            previous = state
            max_weight = max(max_weight, float(weights.max()))
            continue
        asset_return = returns.loc[timestamp].to_numpy(dtype=float)
        growth = 1.0 + float(np.dot(weights, asset_return))
        if growth <= 0:
            raise RuntimeError(f"{timestamp.date()} 组合净值为负")
        before = wealth
        wealth *= growth
        drifted = weights * (1.0 + asset_return) / growth
        if not np.array_equal(state, previous):
            turnover = float(np.abs(target.to_numpy(dtype=float) - drifted).sum())
            wealth *= 1.0 - turnover * (cost_bps / 1e4)
            weights = target.to_numpy(dtype=float)
            rebalances += 1
        else:
            weights = drifted
        previous = state
        max_weight = max(max_weight, float(np.max(weights)))
        rows.append((timestamp, wealth / before - 1.0))
    pnl = pd.Series({timestamp: value for timestamp, value in rows}).sort_index()
    stats = {
        "调仓次数": rebalances,
        "倾斜天数": tilt_days,
        "重叠天数": overlap_days,
        "缩小天数": scaled_days,
        "现金天数": cash_days,
        "最大单品种权重": max_weight,
        "样本日历天数": len(sides),
    }
    return pnl, stats


def run():
    _check_targets()
    out_dir = ROOT / "output" / date.today().isoformat() / "books_tilt_100"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [ticker for ticker in TICKERS if ticker != "KBE"]
    presets = load_presets(names)
    returns, sides, counts = build_frame(names, presets)
    flat = pd.DataFrame(0.0, index=sides.index, columns=sides.columns)
    rows = []
    chart = []
    stats = None
    for cost in COST_BPS:
        tilted, stats = simulate(returns, sides, cost)
        held, _held_stats = simulate(returns, flat, cost)
        rows.append(performance_row("组合", "背离倾斜", cost, tilted))
        rows.append(performance_row("组合", "等权持有", cost, held))
        if cost == 0.0:
            chart = [
                ("背离倾斜", tilted, "-"),
                ("等权持有", held, "k--"),
            ]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot_equity(out_dir / "equity_tilt.png", "背离倾斜 ±100%，不含 KBE，无杠杆", chart)
    print(format_summary(summary).to_string(index=False))
    print()
    print("品种  底背离  顶背离")
    for ticker in names:
        bottom, top = counts[ticker]
        print(f"{ticker:>8}  {bottom:4d}  {top:4d}")
    print()
    for key, value in stats.items():
        if key == "最大单品种权重":
            print(f"{key}: {value * 100:.2f}%")
        else:
            print(f"{key}: {value}")
    print(f"写出 {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
