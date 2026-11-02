"""纯动量，只做多。63 日和 252 日符号取平均，forecast ≤ 0 时空仓。

仓位仍是 forecast × (15% / 品种波动)，上限 2。波动低于目标时权重会大于 1，这是杠杆。
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cta.backtest import (  # noqa: E402
    assert_lagged,
    combine_equal_risk,
    daily_pnl,
    format_summary,
    performance_row,
    plot_equity,
)
from cta.config import COST_BPS, LOOKBACKS, TICKERS  # noqa: E402
from cta.data import load_market  # noqa: E402
from cta.forecast import tsmom_forecast  # noqa: E402
from cta.position import annualized_vol, long_only_forecast, vol_target_weight  # noqa: E402

LONG_NAME = "纯多动量"
SHORT_NAME = "多空动量"


def _exposure(weights):
    frame = pd.DataFrame(weights).sort_index()
    live = frame.stack()
    book = frame.mean(axis=1, skipna=True).dropna()
    return {
        "组合平均敞口": float(book.mean()),
        "组合最大敞口": float(book.max()),
        "权重超过1的占比": float((live > 1.0).mean()),
        "顶到上限2的占比": float((live >= 2.0 - 1e-9).mean()),
        "空仓占比": float((live == 0.0).mean()),
    }


def run():
    if LOOKBACKS != (63, 252):
        raise RuntimeError(f"动量窗口应为 63/252，实际是 {LOOKBACKS}")
    out_dir = ROOT / "output" / date.today().isoformat() / "momentum_long"
    out_dir.mkdir(parents=True, exist_ok=True)

    books = {
        (name, cost): {}
        for name in (LONG_NAME, SHORT_NAME)
        for cost in COST_BPS
    }
    raw = {}
    long_weights = {}
    rows = []

    for ticker in TICKERS:
        daily, _weekly, _meta = load_market(ticker)
        asset_return = daily.Close.pct_change()
        forecast = tsmom_forecast(daily.Close)
        if forecast.iloc[: max(LOOKBACKS)].notna().any():
            raise RuntimeError(f"{ticker} 在预热期内出现了动量预测")
        vol = annualized_vol(daily.Close)
        weights = {
            LONG_NAME: vol_target_weight(long_only_forecast(forecast), vol),
            SHORT_NAME: vol_target_weight(forecast, vol),
        }
        long_held = weights[LONG_NAME].dropna()
        if (long_held < -1e-12).any() or (long_held > 2).any():
            raise RuntimeError(f"{ticker} 纯多仓位超出 [0, 2]")
        long_weights[ticker] = weights[LONG_NAME]
        raw[ticker] = asset_return
        for name, weight in weights.items():
            for cost in COST_BPS:
                pnl = daily_pnl(weight, asset_return, cost)
                if cost == 0:
                    assert_lagged(weight, asset_return, pnl)
                books[(name, cost)][ticker] = pnl
                rows.append(performance_row(ticker, name, cost, pnl, weight))

    chart = []
    for name in (LONG_NAME, SHORT_NAME):
        for cost in COST_BPS:
            combined = combine_equal_risk(books[(name, cost)])
            rows.append(performance_row("组合", name, cost, combined))
            if cost == 0:
                chart.append((name, combined, "-"))
    buyhold = combine_equal_risk(raw).where(chart[0][1].notna())
    rows.append(performance_row("组合", "买入持有", 0.0, buyhold))
    chart.append(("买入持有", buyhold, "k--"))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot_equity(
        out_dir / "equity_long.png",
        "纯多动量，63/252，目标波动 15%，权重上限 2",
        chart,
    )
    shown = format_summary(summary)
    print(shown[shown["标的"] == "组合"].to_string(index=False))
    print()
    print(shown[(shown["预测"] == LONG_NAME) & (shown["成本bps"].isin([0, 0.0]))].to_string(index=False))
    print()
    for key, value in _exposure(long_weights).items():
        print(f"{key}: {value * 100:.2f}%")
    print(f"品种: {', '.join(TICKERS)}")
    print(f"写出 {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
