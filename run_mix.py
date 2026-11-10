"""股票、商品、美元报价外汇放在一个等权组合。不加杠杆。

股票的背离只做底。商品的底背离、顶背离都做。外汇只做动量，不做背离。
纯动量账 forecast ≤ 0 时空仓。切换账里股票和商品的动量可以做空，外汇也用这套动量。
没有保存参数的品种用滤波关、pivot 3、共振 3、去重 10。
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
from cta.divergence_book import peak_holds, switch_position  # noqa: E402
from cta.forecast import tsmom_forecast  # noqa: E402
from cta.mix_book import GROUPS, presets_for  # noqa: E402
from cta.position import long_only_forecast  # noqa: E402
from run_fx import HOLD, MOMENTUM, SWITCH, ensure_daily  # noqa: E402


def run():
    out_dir = ROOT / "output" / date.today().isoformat() / "mix_book_fx_momentum"
    out_dir.mkdir(parents=True, exist_ok=True)
    presets, groups = presets_for()
    names = list(presets)
    markets = {}
    for ticker in names:
        daily, meta = ensure_daily(ticker)
        markets[ticker] = daily
        if groups[ticker] == "外汇":
            hold_text = "不做背离"
            presets[ticker]["底尖峰"] = ""
            presets[ticker]["顶尖峰"] = ""
        else:
            bull_hold, bear_hold = peak_holds(daily, presets[ticker])
            presets[ticker]["底尖峰"] = bull_hold
            presets[ticker]["顶尖峰"] = bear_hold if groups[ticker] == "商品" else ""
            hold_text = f"底 {bull_hold} 根"
            if groups[ticker] == "商品":
                hold_text += f" / 顶 {bear_hold} 根"
        print(
            f"{groups[ticker]}  {ticker}  {meta['日线起始']} → {meta['日线结束']}"
            f"  {presets[ticker]['summary']}  {hold_text}"
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
            momentum_both = forecast.clip(-1, 1)
            if groups[ticker] == "外汇":
                switched = momentum_both
            else:
                switched = switch_position(daily, presets[ticker], momentum_both)
            sleeve = switched.dropna()
            if len(sleeve) and float(sleeve.abs().max()) > 1 + 1e-9:
                raise RuntimeError(f"{ticker} 切换仓位超过 1")
            books[MOMENTUM][ticker] = daily_pnl(momentum, asset_return, cost)
            books[SWITCH][ticker] = daily_pnl(switched, asset_return, cost)
            raw[ticker] = asset_return
            if cost == 0:
                rows.append(performance_row(ticker, SWITCH, cost, books[SWITCH][ticker], switched))
        momentum_book = combine_equal_risk(books[MOMENTUM])
        switch_book = combine_equal_risk(books[SWITCH])
        buyhold = combine_equal_risk(raw).where(momentum_book.notna())
        rows.append(performance_row("组合", MOMENTUM, cost, momentum_book))
        rows.append(performance_row("组合", SWITCH, cost, switch_book))
        if cost == 0:
            rows.append(performance_row("组合", HOLD, cost, buyhold))
            for group, tickers in GROUPS.items():
                rows.append(performance_row(group, SWITCH, cost, combine_equal_risk({t: books[SWITCH][t] for t in tickers})))
            chart = [
                (MOMENTUM, momentum_book, "-"),
                (SWITCH, switch_book, "-"),
                (HOLD, buyhold, "k--"),
            ]

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(
        [{
            "分组": groups[ticker],
            "标的": ticker,
            "规则": preset["summary"],
            "底尖峰": preset.get("底尖峰", ""),
            "顶尖峰": preset.get("顶尖峰", ""),
        } for ticker, preset in presets.items()]
    ).to_csv(out_dir / "presets.csv", index=False)
    plot_equity(out_dir / "equity_mix.png", "外汇只做动量，无杠杆", chart)
    shown = format_summary(summary)
    print()
    print(shown[shown["标的"].isin(["组合", "股票", "商品", "外汇"])].to_string(index=False))
    print()
    print(f"品种数 {len(names)}：股票 {len(GROUPS['股票'])}，商品 {len(GROUPS['商品'])}，外汇 {len(GROUPS['外汇'])}")
    print(f"写出 {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
