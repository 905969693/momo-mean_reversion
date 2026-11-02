"""七个品种各自把背离账和动量账按滚动波动合成。"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cta.backtest import (  # noqa: E402
    combine_by_risk,
    combine_equal_risk,
    daily_pnl,
    format_summary,
    performance_row,
    plot_equity,
    scale_to_target,
)
from cta.config import COST_BPS, TARGET_VOL, TICKERS  # noqa: E402
from cta.data import load_market  # noqa: E402
from cta.divergence_book import divergence_weight, load_presets  # noqa: E402
from cta.forecast import tsmom_forecast  # noqa: E402
from cta.position import annualized_vol, vol_target_weight  # noqa: E402


def _legs(ticker, preset, cost):
    daily, _weekly, _meta = load_market(ticker)
    asset_return = daily.Close.pct_change()
    vol = annualized_vol(daily.Close)
    momentum = daily_pnl(
        vol_target_weight(tsmom_forecast(daily.Close), vol),
        asset_return,
        cost,
    )
    divergence = daily_pnl(divergence_weight(daily, preset), asset_return, cost)
    combined, weights = combine_by_risk({"动量": momentum, "背离": divergence})
    return momentum, divergence, combined, weights, asset_return


def run():
    out_dir = ROOT / "output" / date.today().isoformat() / "books_each"
    out_dir.mkdir(parents=True, exist_ok=True)
    presets = load_presets()
    pd.DataFrame(presets.values()).to_csv(out_dir / "presets.csv", index=False)

    rows = []
    chart = []
    for cost in COST_BPS:
        momentum = {}
        combined = {}
        raw = {}
        for ticker in TICKERS:
            mom, div, both, _weights, asset_return = _legs(ticker, presets[ticker], cost)
            momentum[ticker] = mom
            combined[ticker] = both
            raw[ticker] = asset_return
            rows.append(performance_row(ticker, "动量", cost, mom))
            rows.append(performance_row(ticker, "背离", cost, div))
            rows.append(performance_row(ticker, "两本账", cost, both))
        previous = combine_equal_risk(momentum)
        portfolio = combine_equal_risk(combined)
        scaled = scale_to_target(portfolio, TARGET_VOL)
        buyhold = combine_equal_risk(raw).where(previous.notna())
        rows.append(performance_row("组合", "七品种动量", cost, previous))
        rows.append(performance_row("组合", "七品种两本账", cost, portfolio))
        rows.append(performance_row("组合", "两本账目标15%", cost, scaled))
        if cost == 0:
            rows.append(performance_row("组合", "买入持有", cost, buyhold))
            chart = [
                ("七品种动量", previous, "-"),
                ("两本账", portfolio, "-"),
                ("两本账目标15%", scaled, "-"),
                ("买入持有", buyhold, "k--"),
            ]

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot_equity(out_dir / "equity_books_each.png", "每个品种都用两本账", chart)
    shown = format_summary(summary)
    portfolio_rows = shown[shown["标的"] == "组合"]
    print(portfolio_rows.to_string(index=False))
    print()
    print(f"写出 {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
