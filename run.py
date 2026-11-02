"""读现有日线缓存，跑时间序列动量和均线 ADX 对照，写出净值与汇总。"""
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
from cta.config import (  # noqa: E402
    COST_BPS,
    FORECAST_REGIME,
    FORECAST_TSMOM,
    LOOKBACKS,
    TICKERS,
)
from cta.data import load_market, safe_chart_name  # noqa: E402
from cta.forecast import regime_forecast, tsmom_forecast  # noqa: E402
from cta.position import annualized_vol, vol_target_weight  # noqa: E402


def run():
    out_dir = ROOT / "output" / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    summaries = []
    daily_frames = []
    basket_pnl = {
        (forecast_name, cost): {}
        for forecast_name in (FORECAST_TSMOM, FORECAST_REGIME)
        for cost in COST_BPS
    }
    buyhold_returns = {}

    for ticker in TICKERS:
        daily, _weekly, meta = load_market(ticker)
        sources.append(meta)
        asset_return = daily.Close.pct_change()
        vol = annualized_vol(daily.Close)
        forecasts = {
            FORECAST_TSMOM: tsmom_forecast(daily.Close),
            FORECAST_REGIME: regime_forecast(daily),
        }
        weights = {
            name: vol_target_weight(forecast, vol)
            for name, forecast in forecasts.items()
        }
        _check_warmup(ticker, weights[FORECAST_TSMOM])

        chart_series = []
        for name, weight in weights.items():
            if weight.dropna().empty:
                raise RuntimeError(f"{ticker} 的{name}没有有效仓位")
            held_outside = weight.dropna()
            if ((held_outside < -2) | (held_outside > 2)).any():
                raise RuntimeError(f"{ticker} 的{name}权重超出 [-2, 2]")
            zero_cost = None
            for cost in COST_BPS:
                pnl = daily_pnl(weight, asset_return, cost)
                if cost == 0:
                    assert_lagged(weight, asset_return, pnl)
                    zero_cost = pnl
                basket_pnl[(name, cost)][ticker] = pnl
                summaries.append(performance_row(ticker, name, cost, pnl, weight))
            chart_series.append((name, zero_cost, "-"))
            daily_frames.append(_daily_frame(ticker, name, weight, asset_return, zero_cost, weights_cost5=_pnl_at(weight, asset_return, 5.0)))

        live = chart_series[0][1].notna() | chart_series[1][1].notna()
        buyhold = asset_return.where(live)
        buyhold_returns[ticker] = asset_return
        summaries.append(performance_row(ticker, "买入持有", 0.0, buyhold))
        chart_series.append(("买入持有", buyhold, "k--"))
        plot_equity(
            out_dir / f"equity_{safe_chart_name(ticker)}.png",
            f"{ticker}  趋势跟踪净值",
            chart_series,
        )
        print(
            f"{ticker}  {meta['区间']}  {meta['日线起始']} → {meta['日线结束']}"
            f"  日线 {meta['日线根数']}  周线 {meta['周线根数']}"
        )

    portfolio_buyhold = combine_equal_risk(buyhold_returns)
    basket_chart = []
    for name in (FORECAST_TSMOM, FORECAST_REGIME):
        for cost in COST_BPS:
            combined = combine_equal_risk(basket_pnl[(name, cost)])
            summaries.append(performance_row("组合", name, cost, combined))
            if cost == 0:
                basket_chart.append((name, combined, "-"))
    basket_window = basket_chart[0][1].notna() | basket_chart[1][1].notna()
    portfolio_bh = portfolio_buyhold.where(basket_window)
    summaries.append(performance_row("组合", "买入持有", 0.0, portfolio_bh))
    basket_chart.append(("买入持有", portfolio_bh, "k--"))
    plot_equity(out_dir / "equity_portfolio.png", "等风险组合  趋势跟踪净值", basket_chart)

    summary = pd.DataFrame(summaries)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(sources).to_csv(out_dir / "sources.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(out_dir / "daily.csv", index=False)

    readable = format_summary(summary)
    focus = readable[readable["成本bps"].isin([0, 0.0])]
    print()
    print(focus.to_string(index=False))
    print()
    print(f"写出 {out_dir}")
    return out_dir


def _pnl_at(weight, asset_return, cost_bps):
    return daily_pnl(weight, asset_return, cost_bps)


def _check_warmup(ticker, weight):
    """252 日收益还没形成时，时间序列动量不应有仓位。"""
    warmup = weight.iloc[: max(LOOKBACKS)]
    if warmup.notna().any():
        raise RuntimeError(f"{ticker} 在 {max(LOOKBACKS)} 日预热期内出现了动量仓位")


def _daily_frame(ticker, name, weight, asset_return, pnl_0, weights_cost5):
    frame = pd.DataFrame(
        {
            "日期": weight.index,
            "标的": ticker,
            "预测": name,
            "信号权重": weight.to_numpy(),
            "持仓权重": weight.shift(1).to_numpy(),
            "标的收益": asset_return.to_numpy(),
            "盈亏_0bps": pnl_0.to_numpy(),
            "盈亏_5bps": weights_cost5.to_numpy(),
        }
    )
    return frame.dropna(subset=["持仓权重"])


if __name__ == "__main__":
    run()
