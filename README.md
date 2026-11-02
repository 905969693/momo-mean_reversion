# momo & mean reversion

动量（momo）与均值回归（背离切换）组合回测，以及 Streamlit 持仓看板。

独立于 `tradingview` 背离项目。默认只读 `tradingview/cache` 里已经缓存的日线；看板「更新至最新」会把增量写到本项目的 `cache/`，不写回 tradingview。

## Streamlit 看板

```bash
cd "/Users/yanyunfeng/文稿/cta"
/path/to/.venv/bin/pip install -r requirements.txt
/path/to/.venv/bin/streamlit run streamlit_app.py
```

组合规则：股票只做底背离，商品顶底背离，外汇只做动量；列表可增删标的；无杠杆。

---

# CTA 趋势跟踪（脚本）

周线由同一份日线按周五收盘重采样，未走完的最后一周丢掉。

## 规则

- 时间序列动量：63 / 252 日收益的符号取平均，得到 [-1, 1] 的方向。
- 对照：收盘在 200 日均线之上、ADX(14) > 25 且 +DI 大于 -DI 时为 +1；相反为 -1；其余已形成均线的日子为 0。
- 仓位 = 方向 × (15% / 该品种年化波动)。波动是日收益的指数加权标准差（span 32）。权重截断在 [-2, 2]。
- 当天收盘算出的权重，赚下一根 K 线的收益。5 bps 列按换手扣除单边成本。
- 组合把每个品种已经缩放过的日收益等权平均。当天还在预热的品种不进入分母。

标的：SPY、EEM、KBE、TLT、GLD、USDJPY=X、USO。每个标的在 `2y`、`5y`、`10y`、`max` 里用起始日最早的那份缓存。

`python run_switch.py` 是无杠杆切换，组合不含 KBE：每个品种平时用 63/252 动量方向，仓位在 [-1, 1]；该品种背离确认日收盘起改为背离方向，持有 10 天后回到动量。各品种等权，总敞口不超过本金。结果在 `output/<日期>/books_switch_no_kbe/`。含 KBE 的上一轮留在 `books_switch/`。

两本账有两种跑法。`python run_books.py` 对七个品种各自用自己的背离参数做一账、时间序列动量做一账，按截至昨天的三年滚动波动分配资金，再把七条结果等权合成。背离参数：KBE 用十年回测固定的多空参数，其余读 `fixed_presets.json`。结果在 `output/<日期>/books_each/`。其中「两本账目标15%」会把组合再放大，最高 8 倍，那是杠杆。

## 运行

```bash
cd "/Users/yanyunfeng/文稿/cta"
python run.py
```

依赖见 `requirements.txt`（含 streamlit、yfinance、altair、scipy 等）。部署到 Streamlit Cloud 时无需本机 tradingview 路径；可选环境变量 `CTA_CACHE_DIR` 指定只读缓存目录。

## 输出

`output/<日期>/`

- `equity_<标的>.png`：两条策略与买入持有
- `equity_portfolio.png`：等风险组合
- `summary.csv`：年化收益、波动、夏普、最大回撤、多空占比、平均持仓天数；含 0 bps 与 5 bps
- `sources.csv`：用了哪份日线缓存，以及周线根数
- `daily.csv`：信号权重、滞后一天的持仓权重和盈亏
