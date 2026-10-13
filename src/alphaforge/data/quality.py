"""Data-quality audit of a cleaned panel: what a researcher must know before trusting a backtest.

Cleaning (:mod:`alphaforge.data.cleaning`) *repairs* what it safely can. This module *reports*
what is left, per symbol, and classifies it:

* **errors** can silently bias results and fail the run under ``on_fail: error``:
  ``long_gap`` (interior hole longer than ``max_gap_bars``), ``stale`` (a run of unchanged
  closes on zero volume, i.e. a dead feed), ``funding_gap`` (perp funding history with holes).
* **info** is worth knowing but legitimate in crypto: ``ended_early`` (data stops before the
  panel end: delisting or a dead feed; positions are flattened at the last price, which is
  optimistic for a delisting), ``extreme_move`` (single-bar moves beyond ``spike_mad`` MADs
  that the cleaner did not repair because they did not revert).

Everything here looks at the whole sample at once: it is an audit, never a model input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.config.schema import QualityConfig
from alphaforge.data.market_data import MarketData
from alphaforge.utils.time import timeframe_to_timedelta

ERROR_FLAGS = ("long_gap", "stale", "funding_gap")
INFO_FLAGS = ("ended_early", "extreme_move")


class DataQualityError(ValueError):
    """Raised when ``on_fail: error`` and a symbol has an error-level quality flag."""


@dataclass
class QualityReport:
    """Per-symbol quality table plus the verdict."""

    table: pd.DataFrame  # one row per symbol

    @property
    def errors(self) -> dict[str, list[str]]:
        """``{symbol: [error flags]}`` for symbols with at least one error-level flag."""
        out: dict[str, list[str]] = {}
        for symbol, flags in self.table["flags"].items():
            bad = [f for f in str(flags).split(",") if f in ERROR_FLAGS]
            if bad:
                out[str(symbol)] = bad
        return out

    @property
    def passed(self) -> bool:
        return not self.errors

    def one_line(self) -> str:
        """Compact form for logs, e.g. ``2/12 symbols flagged: FTM(ended_early), ...``."""
        flagged = self.table[self.table["flags"] != ""]
        detail = ", ".join(f"{s}({f})" for s, f in flagged["flags"].items())
        return f"Data quality: {len(flagged)}/{len(self.table)} symbols flagged: {detail}"

    def summary(self) -> str:
        """One line per flagged symbol, for logs and the CLI."""
        flagged = self.table[self.table["flags"] != ""]
        if flagged.empty:
            return f"Data quality: {len(self.table)} symbols, no flags"
        lines = [f"Data quality: {len(flagged)}/{len(self.table)} symbols flagged"]
        for symbol, row in flagged.iterrows():
            lines.append(f"  {symbol}: {row['flags']}")
        return "\n".join(lines)


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True``."""
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _funding_coverage(events: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Share of ``[start, end]`` covered by the venue's own funding cadence.

    The cadence is the median spacing between settlements (8h on Binance, 1h on some venues,
    and it can change per symbol). Any spacing beyond 1.5x the cadence counts as missing time.
    """
    ts = pd.DatetimeIndex(events.index).sort_values()
    ts = ts[(ts >= start) & (ts <= end + pd.Timedelta(days=1))]
    if len(ts) < 3:
        return 0.0
    # Seconds since ``start``: never mix ``asi8`` (resolution-dependent in pandas 3) with ns.
    seconds = (ts - start).total_seconds().to_numpy()
    gaps = np.diff(seconds)
    cadence = float(np.median(gaps))
    missing = float(np.clip(gaps - 1.5 * cadence, 0.0, None).sum())
    lead = max(0.0, float(seconds[0]) - 1.5 * cadence)
    span = (end - start).total_seconds()
    return float(max(0.0, 1.0 - (missing + lead) / span)) if span > 0 else 1.0


def assess_quality(
    data: MarketData,
    config: QualityConfig | None = None,
    funding_events: Mapping[str, pd.Series] | None = None,
) -> QualityReport:
    """Audit a cleaned panel symbol by symbol.

    Args:
        data: Cleaned, aligned panel.
        config: Thresholds; defaults to :class:`QualityConfig`.
        funding_events: Raw funding settlements per symbol (exchange data). When omitted the
            funding check is skipped (synthetic data has no settlement events).
    """
    cfg = config or QualityConfig()
    bar = timeframe_to_timedelta(data.timeframe)
    close, volume = data.close, data.volume
    log_ret = np.log(close / close.shift(1))
    end_of_panel = data.index[-1]
    rows = []
    for symbol in close.columns:
        c = close[symbol]
        valid = c.dropna()
        if valid.empty:
            rows.append({"symbol": symbol, "bars": 0, "flags": "empty"})
            continue
        first, last = valid.index[0], valid.index[-1]
        span = c.loc[first:last]
        interior_nan = span.isna().to_numpy()
        unchanged = (span.diff() == 0) & (volume[symbol].loc[first:last] == 0)
        r = log_ret[symbol].loc[first:last].dropna()
        mad = 1.4826 * float((r - r.median()).abs().median()) if len(r) else 0.0
        n_extreme = int((r.abs() > cfg.spike_mad * mad).sum()) if mad > 0 else 0
        bars_short = round((end_of_panel - last) / bar)
        coverage = float("nan")
        if funding_events is not None and symbol in funding_events:
            coverage = _funding_coverage(funding_events[symbol], first, last)

        flags = []
        longest_gap = _longest_run(interior_nan)
        longest_stale = _longest_run(unchanged.to_numpy())
        if longest_gap > cfg.max_gap_bars:
            flags.append("long_gap")
        if longest_stale >= cfg.stale_bars:
            flags.append("stale")
        if np.isfinite(coverage) and coverage < cfg.min_funding_coverage:
            flags.append("funding_gap")
        if bars_short > cfg.delisting_bars:
            flags.append("ended_early")
        if n_extreme:
            flags.append("extreme_move")
        rows.append(
            {
                "symbol": symbol,
                "first_bar": first,
                "last_bar": last,
                "bars": int(valid.size),
                "interior_nan_bars": int(interior_nan.sum()),
                "longest_gap": longest_gap,
                "longest_stale": longest_stale,
                "zero_volume_share": float((volume[symbol].loc[first:last] == 0).mean()),
                "extreme_moves": n_extreme,
                "funding_coverage": coverage,
                "bars_before_panel_end": bars_short,
                "flags": ",".join(flags),
            }
        )
    return QualityReport(pd.DataFrame(rows).set_index("symbol"))


def enforce_quality(report: QualityReport, on_fail: str) -> None:
    """Apply the ``on_fail`` policy (``error`` raises; ``warn``/``ignore`` are the caller's)."""
    if on_fail == "error" and not report.passed:
        detail = "; ".join(f"{s}: {', '.join(f)}" for s, f in report.errors.items())
        raise DataQualityError(
            f"Data quality check failed ({detail}). Inspect `alphaforge validate <config>`, fix "
            "the data, relax data.quality thresholds, or set data.quality.on_fail: warn"
        )
