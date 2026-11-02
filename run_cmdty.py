"""商品 ETF：GLD、SLV、USO、UNG、DBA、GSG。纯动量、切换、买入持有，不加杠杆。

背离允许多空：底背离做多，顶背离做空。其余过滤沿用已保存参数。
UNG 没有保存参数，用滤波关、pivot 3、共振 3、去重 10，多空都做。
"""
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
from cta.config import COST_BPS  # noqa: E402
from cta.divergence_book import load_presets, switch_position  # noqa: E402
from cta.forecast import tsmom_forecast  # noqa: E402
from cta.position import long_only_forecast  # noqa: E402
from run_fx import HOLD, MOMENTUM, SWITCH, default_preset, ensure_daily  # noqa: E402

NAMES = ["GLD", "SLV", "USO", "UNG", "DBA", "GSG"]
SAVED = ["GLD", "SLV", "USO", "DBA", "GSG"]


def presets_for():
    saved = load_presets(SAVED)
    presets = {}
    for ticker in NAMES:
        preset = dict(saved[ticker]) if ticker in saved else default_preset(ticker)
        preset["long_only"] = False
        preset["summary"] = preset["summary"].replace("、只做多", "") + "、多空都做"
        presets[ticker] = preset
    return presets


def run():
    out_dir = ROOT / "output" / date.today().isoformat() / "cmdty_book_short"
    out_dir.mkdir(parents=True, exist_ok=True)
    presets = presets_for()
    markets = {}
    sources = []
    for ticker in NAMES:
        daily, meta = ensure_daily(ticker)
        markets[ticker] = daily
        sources.append(meta)
        print(
            f"{ticker}  {meta.get('来源', meta.get('区间', ''))}  "
            f"{meta['日线起始']} → {meta['日线结束']}  {meta['日线根数']} 根  {presets[ticker]['summary']}"
        )

    rows = []
    chart = []
    for cost in COST_BPS:
        books = {MOMENTUM: {}, SWITCH: {}}
        raw = {}
        for ticker, daily in markets.items():
            asset_return = daily.Close.pct_change()
            forecast = tsmom_forecast(daily.Close)
            momentum = long_only_forecast(forecast)
            held = momentum.dropna()
            if len(held) and ((held < -1e-12).any() or (held > 1).any()):
                raise RuntimeError(f"{ticker} 纯动量仓位超出 [0, 1]")
            switched = switch_position(daily, presets[ticker], forecast.clip(-1, 1))
            sleeve = switched.dropna()
            if len(sleeve) and float(sleeve.abs().max()) > 1 + 1e-9:
                raise RuntimeError(f"{ticker} 切换仓位超过 1")
            books[MOMENTUM][ticker] = daily_pnl(momentum, asset_return, cost)
            books[SWITCH][ticker] = daily_pnl(switched, asset_return, cost)
            raw[ticker] = asset_return
            if cost == 0:
                rows.append(performance_row(ticker, MOMENTUM, cost, books[MOMENTUM][ticker], momentum))
                rows.append(performance_row(ticker, SWITCH, cost, books[SWITCH][ticker], switched))
        momentum_book = combine_equal_risk(books[MOMENTUM])
        switch_book = combine_equal_risk(books[SWITCH])
        buyhold = combine_equal_risk(raw).where(momentum_book.notna())
        rows.append(performance_row("组合", MOMENTUM, cost, momentum_book))
        rows.append(performance_row("组合", SWITCH, cost, switch_book))
        if cost == 0:
            rows.append(performance_row("组合", HOLD, cost, buyhold))
            chart = [
                (MOMENTUM, momentum_book, "-"),
                (SWITCH, switch_book, "-"),
                (HOLD, buyhold, "k--"),
            ]

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(sources).to_csv(out_dir / "sources.csv", index=False)
    pd.DataFrame(
        [{"标的": ticker, "规则": preset["summary"]} for ticker, preset in presets.items()]
    ).to_csv(out_dir / "presets.csv", index=False)
    plot_equity(out_dir / "equity_cmdty.png", "商品 ETF：背离可做空，无杠杆", chart)
    shown = format_summary(summary)
    print()
    print(shown[shown["标的"] == "组合"].to_string(index=False))
    print()
    print(shown[(shown["标的"] != "组合") & (shown["成本bps"].isin([0, 0.0]))].to_string(index=False))
    print()
    print(f"写出 {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
