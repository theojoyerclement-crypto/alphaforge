"""Point-in-time tradable universe.

Choosing a universe as "today's large caps" is a hindsight bias: those coins are large *because*
they went up. :func:`universe_mask` instead decides membership at each bar from data available
at that bar: a symbol must have traded for ``min_history`` bars and, optionally, rank in the
``top_n`` by trailing dollar volume. Membership is reviewed every ``rebalance_every`` bars and
held in between (avoids churn from daily rank noise).

Limitation, stated where it matters: a mask cannot resurrect coins that the data source no
longer serves. Exchange APIs list live instruments, so delisted perps are absent from the
candidate set; the mask removes *selection* hindsight, not *survivorship* in the data itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.config.schema import UniverseConfig
from alphaforge.data.market_data import MarketData


def universe_mask(data: MarketData, config: UniverseConfig) -> pd.DataFrame:
    """Boolean ``time × symbol`` panel: may symbol ``s`` be held at bar ``t``?

    Row ``t`` uses data up to ``t`` only (trailing listing age and dollar volume), so the mask
    is causal and can gate signals directly.
    """
    listed = data.close.notna()
    eligible = listed & (listed.cumsum() > config.min_history)
    if config.top_n is None:
        return eligible

    adv = data.dollar_volume().rolling(config.rank_lookback, min_periods=1).mean()
    rank = adv.where(eligible).rank(axis=1, ascending=False, method="first")
    reviewed = rank.le(config.top_n) & eligible

    # Hold membership between reviews; a symbol that stops trading drops out immediately.
    values = reviewed.to_numpy(dtype=float, copy=True)
    values[np.arange(len(values)) % config.rebalance_every != 0] = np.nan
    held = pd.DataFrame(values, index=reviewed.index, columns=reviewed.columns).ffill()
    return held.fillna(0.0).astype(bool) & listed
