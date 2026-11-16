"""股票、商品、美元报价外汇放在一个等权组合。不加杠杆。

切换账在动量上覆盖每个标的自己的主方法。没有主方法的方向保持动量。
纯动量账 forecast ≤ 0 时空仓。切换账的动量可以做空。
没有保存参数的品种用滤波关、pivot 3、共振 3、去重 10、只做多。
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
from cta.divergence_book import primary_switch_position  # noqa: E402
from cta.forecast import tsmom_forecast  # noqa: E402
from cta.mix_book import presets_for  # noqa: E402
from cta.universe import listed_members  # noqa: E402
from cta.position import long_only_forecast  # noqa: E402
from run_fx import HOLD, MOMENTUM, SWITCH, ensure_daily  # noqa: E402


def run():
    out_dir = ROOT / "output" / date.today().isoformat() / "mix_book_fx_momentum"
    out_dir.mkdir(parents=True, exist_ok=True)
    presets, groups = presets_for(listed_members())
    names = list(presets)
    markets = {}
    switched_weights = {}
    for ticker in names:
        daily, meta = ensure_daily(ticker)
        markets[ticker] = daily
        weight, _source, method = primary_switch_position(
            daily, presets[ticker], tsmom_forecast(daily.Close).clip(-1, 1),
        )
        switched_weights[ticker] = weight
        presets[ticker]["主方法"] = method
        print(
            f"{groups[ticker]}  {ticker}  {meta['日线起始']} → {meta['日线结束']}"
            f"  {presets[ticker]['summary']}  {method}"
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
            switched = switched_weights[ticker]
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
            by_group = {}
            for ticker, group in groups.items():
                by_group.setdefault(group, {})[ticker] = books[SWITCH][ticker]
            for group in ("股票", "商品", "外汇"):
                sleeves = by_group.get(group)
                if sleeves:
                    rows.append(performance_row(group, SWITCH, cost, combine_equal_risk(sleeves)))
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
    counts = {group: sum(item == group for item in groups.values()) for group in ("股票", "商品", "外汇")}
    print(
        f"品种数 {len(names)}：股票 {counts['股票']}，商品 {counts['商品']}，外汇 {counts['外汇']}"
    )
    print(f"写出 {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
