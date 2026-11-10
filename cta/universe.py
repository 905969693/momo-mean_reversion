"""组合里有哪些标的。默认名单减掉隐藏的，再加上手动加的。

隐藏和新增分别记在 cta/cache/hidden_tickers.txt、extra_tickers.txt。行情不放这里。
只改名单，不删行情缓存。
"""
from pathlib import Path

from cta.mix_book import GROUPS, GROUP_ORDER, default_members

ROOT = Path(__file__).resolve().parents[1]
LIST_DIR = ROOT / "cache"
HIDDEN_PATH = LIST_DIR / "hidden_tickers.txt"
EXTRA_PATH = LIST_DIR / "extra_tickers.txt"


def load_hidden():
    if not HIDDEN_PATH.exists():
        return set()
    return {line.strip() for line in HIDDEN_PATH.read_text().splitlines() if line.strip()}


def save_hidden(names):
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(sorted(names))
    HIDDEN_PATH.write_text(text + ("\n" if text else ""))


def load_extras():
    if not EXTRA_PATH.exists():
        return []
    pairs = []
    for line in EXTRA_PATH.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        ticker, _sep, group = text.partition("\t")
        ticker = ticker.strip()
        group = group.strip()
        if ticker and group in GROUP_ORDER:
            pairs.append((ticker, group))
    return list(dict.fromkeys(pairs))


def save_extras(pairs):
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"{ticker}\t{group}" for ticker, group in dict.fromkeys(pairs)]
    text = "\n".join(lines)
    EXTRA_PATH.write_text(text + ("\n" if text else ""))


def canonical_ticker(name):
    cleaned = name.strip()
    if not cleaned:
        return ""
    for ticker, _group in default_members():
        if cleaned.lower() == ticker.lower():
            return ticker
    return cleaned


def listed_members(hidden=None, extras=None):
    """默认名单里没被隐藏的，再加上手动加入的。手动指定的分组盖过默认分组。"""
    if hidden is None:
        hidden = load_hidden()
    if extras is None:
        extras = load_extras()
    override = {ticker: group for ticker, group in extras}
    members = []
    seen = set()
    for group, tickers in GROUPS.items():
        for ticker in tickers:
            if ticker in hidden:
                continue
            chosen = override.get(ticker, group)
            members.append((ticker, chosen))
            seen.add(ticker)
    for ticker, group in extras:
        if ticker in hidden or ticker in seen:
            continue
        members.append((ticker, group))
        seen.add(ticker)
    return members


def add_member(name, group):
    ticker = canonical_ticker(name)
    if not ticker:
        return []
    if group not in GROUP_ORDER:
        raise ValueError(f"分组必须是股票、商品或外汇，收到 {group}")
    hidden = load_hidden()
    hidden.discard(ticker)
    save_hidden(hidden)
    extras = [(item, item_group) for item, item_group in load_extras() if item != ticker]
    default_group = dict(default_members()).get(ticker)
    if default_group != group:
        extras.append((ticker, group))
    save_extras(extras)
    return listed_members()


def remove_member(name):
    ticker = canonical_ticker(name)
    hidden = load_hidden()
    hidden.add(ticker)
    save_hidden(hidden)
    extras = [(item, group) for item, group in load_extras() if item != ticker]
    save_extras(extras)
    return listed_members()
