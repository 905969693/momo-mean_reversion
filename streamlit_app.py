"""当前组合的持仓，以及最近 30 个交易日的调仓。"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cta.mix_book import GROUP_ORDER, build_book, slice_pnl_history
from cta.universe import add_member, listed_members, remove_member
from run_fx import portfolio_as_of, refresh_listed

WINDOW = 30
# build_book 返回字段有变时加 1，避免 st.cache_data 继续用旧结果。
BOOK_CACHE_VERSION = 3

PNL_SPANS = {
    "1 个月": 21,
    "3 个月": 63,
    "6 个月": 126,
    "1 年": 252,
    "2 年": 504,
    "全部": None,
}


@st.cache_data(ttl="30m", show_spinner="正在读取行情并计算持仓")
def load_book(members, data_rev, book_cache_version):
    del data_rev, book_cache_version
    return build_book(members, sessions=WINDOW)


def _subset(frame, group, held_only):
    if frame is None or frame.empty:
        return frame
    view = frame
    if group != "全部":
        view = view[view["分组"] == group]
    if held_only and "方向" in view.columns:
        view = view[view["方向"] != "空仓"]
    return view


def _metrics(snapshot, changes):
    if snapshot is None or snapshot.empty:
        net = long = short = 0.0
        held = 0
        count = 0
    else:
        net = float(snapshot["组合权重"].sum())
        long = float(snapshot["组合权重"].clip(lower=0).sum())
        short = float(snapshot["组合权重"].clip(upper=0).sum())
        held = int((snapshot["方向"] != "空仓").sum())
        count = len(snapshot)
    trade_count = 0 if changes is None or changes.empty else len(changes)
    with st.container(horizontal=True):
        st.metric(
            "净敞口",
            f"{net:.1%}",
            help="多头组合权重减去空头组合权重。筛选分组时，只加总该分组。",
            border=True,
        )
        st.metric("多头敞口", f"{long:.1%}", border=True)
        st.metric("空头敞口", f"{short:.1%}", border=True)
        st.metric("有仓位", f"{held} / {count}", help="方向不是空仓的品种数。", border=True)
        st.metric("调仓", f"{trade_count} 笔", help="最近 30 个交易日里，品种仓位发生变化的次数。", border=True)


def _pnl_compare_chart(frame):
    plot = frame[["日期", "净值", "SPY"]].rename(columns={"净值": "组合", "SPY": "SPY"})
    long = plot.melt(id_vars="日期", var_name="系列", value_name="指数")
    long = long.dropna(subset=["指数"])
    if long.empty:
        return
    lo = float(long["指数"].min())
    hi = float(long["指数"].max())
    pad = max((hi - lo) * 0.06, 0.015)
    y_scale = alt.Scale(domain=[lo - pad, hi + pad], zero=False, nice=False)
    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("日期:T", title="日期"),
            y=alt.Y("指数:Q", title="净值", scale=y_scale, axis=alt.Axis(format=".2f")),
            color=alt.Color(
                "系列:N",
                title="系列",
                scale=alt.Scale(
                    domain=["组合", "SPY"],
                    range=["#34D399", "#60A5FA"],
                ),
            ),
            tooltip=[
                alt.Tooltip("日期:T", title="日期"),
                alt.Tooltip("系列:N", title="系列"),
                alt.Tooltip("指数:Q", title="净值", format=".3f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, alt="过去一年组合净值与 SPY 买入持有对比")


def _holdings_chart(snapshot):
    order = list(snapshot["标的"])
    chart = (
        alt.Chart(snapshot)
        .mark_bar()
        .encode(
            x=alt.X("组合权重:Q", title="组合权重", axis=alt.Axis(format="%")),
            y=alt.Y("标的:N", title=None, sort=order),
            color=alt.Color(
                "方向:N",
                title="方向",
                scale=alt.Scale(
                    domain=["多", "空", "空仓"],
                    range=["#34D399", "#F87171", "#94A3B8"],
                ),
            ),
            tooltip=[
                alt.Tooltip("标的:N", title="标的"),
                alt.Tooltip("分组:N", title="分组"),
                alt.Tooltip("方向:N", title="方向"),
                alt.Tooltip("组合权重:Q", title="组合权重", format=".2%"),
                alt.Tooltip("品种仓位:Q", title="品种仓位", format=".0%"),
                alt.Tooltip("来源:N", title="来源"),
            ],
        )
        .properties(height=max(280, 28 * len(snapshot)))
    )
    st.altair_chart(chart, alt="各品种占组合的权重，多头在右侧，空头在左侧")


def _weight_columns():
    return {
        "品种仓位": st.column_config.NumberColumn(
            "品种仓位",
            format="percent",
            help="这一份品种资金的方向。1 是多满，-1 是空满。",
        ),
        "组合权重": st.column_config.NumberColumn(
            "组合权重",
            format="percent",
            help="占整个账户的比例。品种之间等权，没有杠杆。",
        ),
        "原仓位": st.column_config.NumberColumn("原仓位", format="percent"),
        "新仓位": st.column_config.NumberColumn("新仓位", format="percent"),
        "原组合权重": st.column_config.NumberColumn("原组合权重", format="percent"),
        "新组合权重": st.column_config.NumberColumn("新组合权重", format="percent"),
        "收盘": st.column_config.NumberColumn("收盘", format="%.4f"),
        "标的": st.column_config.TextColumn("标的", pinned=True),
    }


def _render_ticker_menu():
    members = listed_members()
    with st.popover(
        f"{len(members)} 个标的" if members else "添加标的",
        width="stretch",
        help="添加或删除组合里的标的。删除只从这张持仓拿掉，不删行情缓存。",
    ):
        st.caption("股票只做底背离，商品顶底都做，外汇只做动量。新加的标的按这里选的分组走。")
        new_name = st.text_input(
            "添加标的",
            placeholder="SPY、GLD、USDJPY=X",
            key="new_ticker_input",
        )
        new_group = st.segmented_control(
            "分组",
            list(GROUP_ORDER),
            default="股票",
            key="new_ticker_group",
        )
        if st.button("添加", key="add_ticker_btn", width="stretch"):
            name = new_name.strip()
            if name:
                add_member(name, new_group or "股票")
                load_book.clear()
                st.rerun()
        if not members:
            st.caption("列表是空的。")
        for ticker, group in members:
            label, remove = st.columns([4, 1], vertical_alignment="center")
            label.markdown(f"**{ticker}** · {group}")
            if remove.button("删除", key=f"del_ticker_{ticker}", help=f"从组合移除 {ticker}"):
                remove_member(ticker)
                load_book.clear()
                st.rerun()
    return members


st.set_page_config(page_title="组合持仓", layout="wide")
st.title("组合持仓")
st.caption("每个品种一份等权资金，仓位在 -1 到 1，不加杠杆。收盘上的仓位从下一交易日开始计收益。")

if "data_rev" not in st.session_state:
    st.session_state.data_rev = 0

with st.sidebar:
    status_slot = st.container()
    st.header("组合")
    members = _render_ticker_menu()
    st.caption("调仓只记录品种仓位的变化。每天的等权再平衡不单列。")
    st.divider()
    st.header("筛选")
    group = st.segmented_control(
        "分组",
        ["全部", "股票", "商品", "外汇"],
        default="全部",
        key="view_group",
    )
    held_only = st.toggle("只看有仓位", value=False)
    if st.button("重新计算", icon=":material/refresh:", width="stretch"):
        load_book.clear()
        st.rerun()

with status_slot:
    quote = portfolio_as_of([ticker for ticker, _group in members])
    if quote["earliest"] is None:
        st.badge(
            "数据尚未下载",
            icon=":material/calendar_today:",
            color="orange",
            width="stretch",
            help="列表里的标的还没有本地日线。",
        )
    else:
        help_text = f"列表里最后一根最早停在 {quote['earliest_ticker']}。"
        if quote["latest"] != quote["earliest"]:
            help_text += f" 最晚的标的到 {quote['latest']}。"
        if quote["missing"]:
            help_text += " 还没有缓存：" + "、".join(quote["missing"]) + "。"
        st.badge(
            f"数据更新至 {quote['earliest']}",
            icon=":material/calendar_today:",
            color="gray",
            width="stretch",
            help=help_text,
        )
    if st.button(
        "更新至最新",
        key="update_to_latest",
        icon=":material/download:",
        width="stretch",
        help="列表里的标的都会补到最新。已有的 K 线会保留，只下载还没有的交易日。",
    ):
        refresh_errors = refresh_listed([ticker for ticker, _group in members], status_slot)
        st.session_state.price_refresh_errors = refresh_errors
        st.session_state.data_rev += 1
        load_book.clear()
        st.rerun()
    refresh_errors = st.session_state.pop("price_refresh_errors", None)
    if refresh_errors is not None:
        if refresh_errors:
            st.caption("部分标的没有更新：" + "；".join(refresh_errors))
        else:
            st.caption("已补到最新。")

if not members:
    st.warning("列表是空的，请先在左侧添加一个标的。", title="没有标的")
    st.stop()

group = group or "全部"
book = load_book(tuple(members), st.session_state.data_rev, BOOK_CACHE_VERSION)
if book["failed"]:
    st.warning("；".join(book["failed"]), title="部分标的没有行情")

snapshot = _subset(book["snapshot"], group, held_only)
changes = _subset(book["changes"], group, held_only=False)
if book["book_date"]:
    quote_note = ""
    if book["latest_quote"] and book["latest_quote"] != book["book_date"]:
        quote_note = f" 部分品种的行情到 {book['latest_quote']}，没有单独往后推。"
    st.caption(f"持仓对齐到 {book['book_date']}。这是多数品种共同的最近交易日。{quote_note}")

_metrics(snapshot, changes)

with st.container(border=True):
    st.subheader("组合盈亏")
    pnl_returns = book.get("pnl_returns")
    if pnl_returns is None:
        st.info("缓存还是旧版本。点侧边栏「重新计算」后再看。", title="需要刷新")
    elif pnl_returns.empty:
        st.info(
            "组合日盈亏还没有形成。常见原因是列表里的品种都还在动量预热期，"
            "或当前列表没有成功读到行情。",
            title="没有记录",
        )
    else:
        span_label = st.segmented_control(
            "区间",
            list(PNL_SPANS),
            default="1 年",
            key="pnl_span",
        )
        span_label = span_label or "1 年"
        pnl_view = slice_pnl_history(pnl_returns, trading_days=PNL_SPANS[span_label])
        start = pd.Timestamp(pnl_view["日期"].iloc[0]).date().isoformat()
        end = pd.Timestamp(pnl_view["日期"].iloc[-1]).date().isoformat()
        port_total = float(pnl_view["净值"].iloc[-1] - 1.0)
        spy_total = float(pnl_view["SPY"].iloc[-1] - 1.0) if pnl_view["SPY"].notna().any() else None
        spy_note = f" SPY 同期 {spy_total:.1%}。" if spy_total is not None and np.isfinite(spy_total) else ""
        st.caption(
            f"{span_label}：{start} 至 {end}，{len(pnl_view)} 个交易日。"
            f"组合累计 {port_total:.1%}。{spy_note}"
            "切换策略、品种等权、0 bps；SPY 为买入持有，所选区间起点都归一化为 1。"
        )
        _pnl_compare_chart(pnl_view)

left, right = st.columns([3, 2])
with left:
    with st.container(border=True):
        st.subheader("当前持仓")
        if snapshot is None or snapshot.empty:
            st.info("这个筛选下没有持仓。", title="没有记录")
        else:
            st.dataframe(
                snapshot,
                column_config=_weight_columns(),
                hide_index=True,
                alt="当前每个品种的方向、组合权重和信号来源",
            )
with right:
    with st.container(border=True):
        st.subheader("组合权重")
        if snapshot is None or snapshot.empty:
            st.info("这个筛选下没有可以画的权重。", title="没有记录")
        else:
            _holdings_chart(snapshot)

with st.container(border=True):
    st.subheader("过去 30 个交易日的调仓")
    if book["window_start"]:
        st.caption(
            f"{book['window_start']} 至 {book['window_end']}。"
            "日期是新仓位生效的收盘。原仓位赚到这一天，新仓位从下一交易日开始赚。"
        )
    if changes is None or changes.empty:
        st.info("这 30 个交易日里没有品种改变仓位。", title="没有调仓")
    else:
        st.dataframe(
            changes,
            column_config=_weight_columns(),
            hide_index=True,
            alt="最近 30 个交易日里品种仓位发生变化的记录",
        )

with st.container(border=True):
    st.subheader("净敞口")
    if group != "全部":
        st.caption("净敞口是整个组合的，不随分组筛选改变。")
    exposure = book["exposure"]
    if exposure is None or exposure.empty:
        st.info("还没有足够的行情来画净敞口。", title="没有记录")
    else:
        st.line_chart(
            exposure,
            x="日期",
            y="净敞口",
            x_label="日期",
            y_label="净敞口",
            alt="最近 30 个交易日组合净敞口",
        )
