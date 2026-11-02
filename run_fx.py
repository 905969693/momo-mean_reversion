"""G10 与即期新兴市场外汇，不含 USDTRY。纯动量、切换、买入持有，三本账都不加杠杆。

没有保存参数的品种用项目默认值：滤波关、pivot 3、共振 3、去重 10、只做多。
已有缓存的品种只读 tradingview/cache，缺的日线写到 cta/cache，不回写原缓存。
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

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
from cta.data import load_market, read_ohlcv, safe_chart_name  # noqa: E402
from cta.divergence_book import HOLD_DAYS, load_presets, switch_position  # noqa: E402
from cta.forecast import tsmom_forecast  # noqa: E402
from cta.position import long_only_forecast  # noqa: E402

LOCAL_CACHE = ROOT / "cache"
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


def _flatten_download(frame):
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    keep = [col for col in ("Open", "High", "Low", "Close", "Volume") if col in frame.columns]
    out = frame.loc[:, keep].copy()
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _local_path(ticker):
    return LOCAL_CACHE / f"{safe_chart_name(ticker)}_10y.csv"


def _adjust_path(ticker):
    return LOCAL_CACHE / f"{safe_chart_name(ticker)}_adjust.txt"


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
    """这份组合实际会用的最后一根。cta 缓存更晚时用它，否则用原缓存。"""
    local = _csv_span(_local_path(ticker))
    local_end = None if local is None else local[1]
    tradingview_end = _tradingview_end(ticker)
    if local_end and (tradingview_end is None or local_end >= tradingview_end):
        return local_end
    return tradingview_end


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


def _read_adjust(ticker, default):
    path = _adjust_path(ticker)
    if not path.is_file():
        return default
    return path.read_text().strip() == "true"


def _write_adjust(ticker, adjust):
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    _adjust_path(ticker).write_text("true\n" if adjust else "false\n")


def _prefer_local(local_df, tradingview_df):
    if local_df is None or local_df.empty:
        return False
    if tradingview_df is None or tradingview_df.empty:
        return True
    local_end = pd.Timestamp(local_df.index[-1]).normalize()
    tradingview_end = pd.Timestamp(tradingview_df.index[-1]).normalize()
    if local_end > tradingview_end:
        return True
    if local_end < tradingview_end:
        return False
    return len(local_df) >= len(tradingview_df)


def _daily_meta(ticker, daily, source, path):
    return {
        "标的": ticker,
        "区间": "10y",
        "缓存文件": str(path),
        "来源": source,
        "日线起始": daily.index[0].date().isoformat(),
        "日线结束": daily.index[-1].date().isoformat(),
        "日线根数": len(daily),
    }


def ensure_daily(ticker):
    tradingview_df = None
    tradingview_meta = None
    try:
        tradingview_df, _weekly, tradingview_meta = load_market(ticker)
    except FileNotFoundError:
        tradingview_df = None
    path = _local_path(ticker)
    local_df = read_ohlcv(path) if path.is_file() else None
    if _prefer_local(local_df, tradingview_df):
        return local_df, _daily_meta(ticker, local_df, "cta/cache", path)
    if tradingview_df is not None:
        tradingview_meta["来源"] = "tradingview/cache"
        return tradingview_df, tradingview_meta
    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    downloaded = yf.download(ticker, period="10y", auto_adjust=False, progress=False)
    flat = _flatten_download(downloaded)
    if flat.empty:
        raise RuntimeError(f"{ticker} 没有下载到日线")
    flat.to_csv(path)
    _write_adjust(ticker, False)
    return flat, _daily_meta(ticker, flat, "cta/cache", path)


def refresh_daily(ticker):
    """只补最后一根之后的交易日，写到 cta/cache。不改 tradingview/cache。

    原缓存的价格是复权的，沿用复权。只存在于 cta/cache 的价格不复权，沿用不复权。
    """
    tradingview_df = None
    try:
        tradingview_df, _weekly, _meta = load_market(ticker)
    except FileNotFoundError:
        tradingview_df = None
    path = _local_path(ticker)
    local_df = read_ohlcv(path) if path.is_file() else None
    if _prefer_local(local_df, tradingview_df):
        base = local_df
        adjust = _read_adjust(ticker, default=False)
    elif tradingview_df is not None:
        base = tradingview_df
        adjust = True
    else:
        base = None
        adjust = False
    if base is None:
        downloaded = yf.download(ticker, period="10y", auto_adjust=adjust, progress=False)
        flat = _flatten_download(downloaded)
        if flat.empty:
            raise RuntimeError(f"{ticker} 没有下载到日线")
        LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
        flat.to_csv(path)
        _write_adjust(ticker, adjust)
        return pd.Timestamp(flat.index[-1]).date().isoformat()
    last = pd.Timestamp(base.index[-1]).normalize()
    today = pd.Timestamp.today().normalize()
    updated = base
    if last + pd.Timedelta(days=1) <= today:
        new_raw = yf.download(
            ticker,
            start=(last + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=adjust,
            progress=False,
        )
        new = _flatten_download(new_raw) if new_raw is not None and not new_raw.empty else pd.DataFrame()
        if not new.empty:
            updated = pd.concat([base, new])
            updated = updated[~updated.index.duplicated(keep="last")].sort_index()
    if updated is not base:
        LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
        updated.to_csv(path)
        _write_adjust(ticker, adjust)
    return pd.Timestamp(updated.index[-1]).date().isoformat()


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
