"""G10 与即期新兴市场外汇，不含 USDTRY。纯动量、切换、买入持有，三本账都不加杠杆。

没有保存参数的品种用项目默认值：滤波关、pivot 3、共振 3、去重 10、只做多。
日线只读、只写 tradingview/cache，和背离看板同一份。
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
from cta.config import CACHE_DIR, COST_BPS, PERIOD_CANDIDATES  # noqa: E402
from cta.data import load_market, safe_chart_name  # noqa: E402
from cta.divergence_book import HOLD_DAYS, _tradingview, load_presets, switch_position  # noqa: E402
from cta.forecast import tsmom_forecast  # noqa: E402
from cta.position import long_only_forecast  # noqa: E402

SAVED = ("USDJPY=X", "USDCHF=X", "USDMXN=X")
NAMES = [
    "EURUSD=X",
    "USDJPY=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "NZDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "USDSEK=X",
    "USDNOK=X",
    "USDMXN=X",
    "USDZAR=X",
    "USDPLN=X",
    "USDHUF=X",
    "USDCZK=X",
    "USDILS=X",
]
MOMENTUM = "纯动量"
SWITCH = "切换"
HOLD = "买入持有"


def default_preset(ticker):
    return {
        "ticker": ticker,
        "pivot_order": 3,
        "min_count": 3,
        "dedup_bars": 10,
        "ma_n": 200,
        "adx_th": None,
        "mode": "off",
        "long_only": True,
        "hold": HOLD_DAYS,
        "summary": "滤波关、pivot 3、共振 3、去重 10、只做多",
    }


def _csv_span(path):
    """返回 (起始日, 结束日, 数据行数)。没有数据行时返回 None。"""
    if not path.is_file():
        return None
    start = end = None
    rows = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            token = line.split(",", 1)[0].strip()
            if token.lower() in {"date", "datetime"}:
                continue
            if start is None:
                start = token[:10]
            end = token[:10]
            rows += 1
    if start is None or end is None:
        return None
    return start, end, rows


def _tradingview_end(ticker):
    """与 load_market 同一份日线的最后一天：起始最早，起始相同则行数更多。"""
    name = safe_chart_name(ticker)
    best = None
    for period in PERIOD_CANDIDATES:
        span = _csv_span(CACHE_DIR / f"{name}_{period}.csv")
        if span is None:
            continue
        rank = (span[0], -span[2])
        if best is None or rank < best[0]:
            best = (rank, span[1])
    return None if best is None else best[1]


def series_as_of(ticker):
    """这份组合实际会用的最后一根，也就是共享日线缓存里会被选中的那份。"""
    return _tradingview_end(ticker)


def portfolio_as_of(tickers):
    """列表里最早的最后一根，以及最晚的最后一根。缺缓存的标的单独列出。"""
    found = []
    missing = []
    for ticker in tickers:
        day = series_as_of(ticker)
        if day is None:
            missing.append(ticker)
        else:
            found.append((ticker, day))
    if not found:
        return {"earliest": None, "latest": None, "earliest_ticker": None, "missing": missing}
    earliest = min(found, key=lambda item: item[1])
    latest = max(found, key=lambda item: item[1])
    return {
        "earliest": earliest[1],
        "latest": latest[1],
        "earliest_ticker": earliest[0],
        "missing": missing,
    }


def _periods_to_refresh(ticker):
    name = safe_chart_name(ticker)
    periods = [
        period for period in PERIOD_CANDIDATES
        if (CACHE_DIR / f"{name}_{period}.csv").is_file()
    ]
    return periods or ["10y"]


def ensure_daily(ticker):
    """读共享日线。还没有缓存时，按 10 年下载进同一份缓存。"""
    try:
        daily, _weekly, meta = load_market(ticker)
    except FileNotFoundError:
        refresh_daily(ticker)
        daily, _weekly, meta = load_market(ticker)
    meta["来源"] = str(CACHE_DIR)
    return daily, meta


def refresh_daily(ticker):
    """增量补齐共享缓存里已有的区间。没有文件时下载 10 年。"""
    tv = _tradingview()
    if Path(tv.CACHE_DIR).resolve() != Path(CACHE_DIR).resolve():
        raise RuntimeError(
            f"信号模块的缓存是 {tv.CACHE_DIR}，CTA 读的是 {CACHE_DIR}，没有写同一份日线。"
        )
    last = None
    for period in _periods_to_refresh(ticker):
        df = tv.load_ohlcv(ticker, period=period)
        stamp = pd.Timestamp(df.index[-1]).date().isoformat()
        if last is None or stamp > last:
            last = stamp
    if last is None:
        raise RuntimeError(f"{ticker} 没有下载到日线")
    return last


def refresh_listed(tickers, parent):
    """按列表补到最新。返回失败说明，空列表表示全部成功。"""
    ordered = list(dict.fromkeys(ticker for ticker in tickers if ticker))
    errors = []
    total = len(ordered)
    progress = parent.progress(0.0, text="准备更新")
    for index, ticker in enumerate(ordered, start=1):
        progress.progress((index - 1) / total if total else 1.0, text=f"正在更新 {ticker}（{index}/{total}）")
        try:
            refresh_daily(ticker)
        except Exception as exc:
            errors.append(f"{ticker}（{exc}）")
    progress.progress(1.0, text="已补到最新" if not errors else "更新结束")
    return errors


def presets_for(names):
    saved = load_presets(list(SAVED))
    presets = {}
    for ticker in names:
        if ticker in saved:
            preset = dict(saved[ticker])
        else:
            preset = default_preset(ticker)
        preset["long_only"] = True
        presets[ticker] = preset
    return presets


def run():
    out_dir = ROOT / "output" / date.today().isoformat() / "fx_book_no_try"
    out_dir.mkdir(parents=True, exist_ok=True)
    presets = presets_for(NAMES)
    markets = {}
    sources = []
    for ticker in NAMES:
        daily, meta = ensure_daily(ticker)
        markets[ticker] = daily
        sources.append(meta)
        print(f"{ticker}  {meta['来源']}  {meta['日线起始']} → {meta['日线结束']}  {meta['日线根数']} 根")

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
    plot_equity(out_dir / "equity_fx.png", "外汇：纯动量、切换、买入持有，无杠杆", chart)
    shown = format_summary(summary)
    print()
    print(shown[shown["标的"] == "组合"].to_string(index=False))
    print()
    print(f"写出 {out_dir}")
    return out_dir


if __name__ == "__main__":
    run()
