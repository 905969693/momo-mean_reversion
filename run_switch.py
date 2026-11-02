"""每个品种平时用动量；背离确认后该品种全仓改做背离，持有 10 天再回到动量。仓位不超过 1。"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cta.backtest import (  # noqa: E402
    combine_equal_risk,
    daily_pnl,
    format_summary,
    performance_row,
    plot_equity,
)
from cta.config import COST_BPS, TICKERS  # noqa: E402
from cta.data import load_market  # noqa: E402
from cta.divergence_book import divergence_only_position, load_presets, switch_position  # noqa: E402
from cta.forecast import tsmom_forecast  # noqa: E402
from cta.position import annualized_vol, vol_target_weight  # noqa: E402


def _sleeve(ticker, preset, cost):
    daily, _weekly, _meta = load_market(ticker)
    asset_return = daily.Close.pct_change()
    momentum = tsmom_forecast(daily.Close).clip(-1, 1)
    bottom_only = dict(preset)
    bottom_only["long_only"] = True
    switched = switch_position(daily, bottom_only, momentum)
    divergence_only = divergence_only_position(daily, bottom_only, both_sides=False)
    levered = vol_target_weight(tsmom_forecast(daily.Close), annualized_vol(daily.Close))
    return {
        "切换": daily_pnl(switched, asset_return, cost),
        "只做底背离": daily_pnl(divergence_only, asset_return, cost),
        "动量无杠杆": daily_pnl(momentum, asset_return, cost),
        "动量含杠杆": daily_pnl(levered, asset_return, cost),
        "收益": asset_return,
    }


def run():
    out_dir = ROOT / "output" / date.today().isoformat() / "books_switch_bottom"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [ticker for ticker in TICKERS if ticker != "KBE"]
    presets = load_presets(names)
    rows = []
    chart = []
    for cost in COST_BPS:
        sleeves = {name: {} for name in ("切换", "只做底背离", "动量无杠杆", "动量含杠杆")}
        raw = {}
        for ticker in names:
            result = _sleeve(ticker, presets[ticker], cost)
            raw[ticker] = result["收益"]
            for name in sleeves:
                sleeves[name][ticker] = result[name]
                rows.append(performance_row(ticker, name, cost, result[name]))
        switched = combine_equal_risk(sleeves["切换"])
        divergence_only = combine_equal_risk(sleeves["只做底背离"])
        plain = combine_equal_risk(sleeves["动量无杠杆"])
        levered = combine_equal_risk(sleeves["动量含杠杆"])
        buyhold = combine_equal_risk(raw).where(plain.notna())
        rows.append(performance_row("组合", "切换", cost, switched))
        rows.append(performance_row("组合", "只做底背离", cost, divergence_only))
        rows.append(performance_row("组合", "动量无杠杆", cost, plain))
        rows.append(performance_row("组合", "动量含杠杆", cost, levered))
        if cost == 0:
            rows.append(performance_row("组合", "买入持有", cost, buyhold))
            chart = [
                ("切换", switched, "-"),
                ("只做底背离", divergence_only, "-"),
                ("动量无杠杆", plain, "-"),
                ("买入持有", buyhold, "k--"),
            ]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot_equity(out_dir / "equity_switch.png", "动量 + 只做底背离，不含 KBE", chart)
    shown = format_summary(summary)
    print(shown[shown["标的"] == "组合"].to_string(index=False))
    print()
    print(shown[(shown["预测"] == "切换") & (shown["成本bps"].isin([0, 0.0]))].to_string(index=False))
    print()
    print(f"写出 {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
