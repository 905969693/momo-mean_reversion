"""按已实现波动把方向缩放成仓位。"""
import numpy as np

from cta.config import ANNUALIZATION, TARGET_VOL, VOL_SPAN, WEIGHT_CAP


def annualized_vol(close, span=VOL_SPAN, annualization=ANNUALIZATION):
    """日收益的指数加权标准差，年化。只用当天及以前的收益。"""
    returns = close.pct_change()
    vol = returns.ewm(span=span, min_periods=span, adjust=False).std()
    return vol * np.sqrt(annualization)


def long_only_forecast(forecast):
    """forecast ≤ 0 时空仓。缺失的预热期仍然缺失，不当成已经决定的空仓。"""
    return forecast.clip(lower=0).where(forecast.notna())


def vol_target_weight(forecast, vol, target=TARGET_VOL, cap=WEIGHT_CAP):
    """权重 = forecast × (目标波动 / 品种波动)，截断到 [-cap, cap]。

    预热期（forecast 或波动还没有）保持缺失，不记成一张已经决定的空仓。
    """
    tradable_vol = vol.where(vol > 0)
    scaled = forecast * (target / tradable_vol)
    weight = scaled.clip(-cap, cap)
    return weight.where(forecast.notna() & tradable_vol.notna())
