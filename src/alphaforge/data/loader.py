"""Data pipeline entry point: config -> raw data -> cleaned, aligned ``MarketData``.

:func:`load_dataset` returns the panel together with what a reviewer needs to trust it: the
cleaning report, the quality audit and the provenance (hash, range, fetch metadata) of every
stored file that was read. :func:`load_market_data` is the original, panel-only entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

import pandas as pd

from alphaforge.config.schema import CleaningConfig, DataConfig
from alphaforge.data.cleaning import CleaningReport, aggregate_funding_to_bars, clean_ohlcv
from alphaforge.data.market_data import MarketData
from alphaforge.data.quality import QualityReport, assess_quality, enforce_quality
from alphaforge.data.sources.synthetic import generate_synthetic_market
from alphaforge.data.storage import ParquetStore
from alphaforge.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Dataset:
    """A loaded panel plus its audit trail."""

    data: MarketData
    cleaning: list[CleaningReport]
    quality: QualityReport
    sources: list[dict[str, Any]] = field(default_factory=list)  # sidecars of the files read


def _clean_all(
    frames: dict[str, pd.DataFrame], timeframe: str, cleaning: CleaningConfig
) -> tuple[dict[str, pd.DataFrame], list[CleaningReport]]:
    cleaned: dict[str, pd.DataFrame] = {}
    reports: list[CleaningReport] = []
    for symbol, frame in frames.items():
        clean, report = clean_ohlcv(frame, timeframe, cleaning, symbol)
        if clean.empty:
            logger.warning("Dropping %s: no valid bars after cleaning", symbol)
            continue
        cleaned[symbol] = clean
        reports.append(report)
        if report.outliers_repaired or report.gaps_remaining or report.duplicates_removed:
            logger.info("Cleaning %s: %s", symbol, report.to_dict())
    return cleaned, reports


def _fetch_meta(cfg: DataConfig) -> dict[str, Any]:
    try:
        client: str | None = f"ccxt {metadata.version('ccxt')}"
    except metadata.PackageNotFoundError:
        client = None
    return {
        "exchange": cfg.exchange,
        "market_type": cfg.market_type,
        "client": client,
        "requested_start": cfg.start,
        "requested_end": cfg.end,
    }


def _load_exchange(
    cfg: DataConfig, store: ParquetStore, use_network: bool
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], list[dict[str, Any]]]:
    source_name = f"ccxt_{cfg.exchange}_{cfg.market_type}"
    client = None
    if use_network:
        from alphaforge.data.sources.ccxt_source import CCXTSource

        client = CCXTSource(cfg.exchange, cfg.market_type)

    frames: dict[str, pd.DataFrame] = {}
    funding_events: dict[str, pd.Series] = {}
    sources: list[dict[str, Any]] = []
    for symbol in cfg.symbols:
        cached = store.exists(source_name, symbol, cfg.timeframe)
        if client is not None and not (cfg.use_cache and cached):
            meta = _fetch_meta(cfg)
            ohlcv = client.fetch_ohlcv(symbol, cfg.timeframe, cfg.start, cfg.end)
            store.write(ohlcv, source_name, symbol, cfg.timeframe, meta=meta)
            if cfg.include_funding and cfg.market_type == "swap":
                events = client.fetch_funding(symbol, cfg.start, cfg.end)
                if events is not None:
                    frame = events.to_frame("funding_rate")
                    store.write(frame, source_name, symbol, "events", "funding", meta=meta)
        if not store.exists(source_name, symbol, cfg.timeframe):
            raise FileNotFoundError(
                f"No stored {cfg.timeframe} data for {symbol} under {store.root / source_name}. "
                f"Download it first (alphaforge fetch --exchange {cfg.exchange} --market-type "
                f"{cfg.market_type} --symbols {' '.join(cfg.symbols)} --timeframe "
                f"{cfg.timeframe} --store {cfg.storage_dir}) or set data.source: ccxt"
            )
        frames[symbol] = store.read(
            source_name, symbol, cfg.timeframe, start=cfg.start, end=cfg.end
        )
        sources.append(store.metadata(source_name, symbol, cfg.timeframe) or {"symbol": symbol})
        if cfg.include_funding and store.exists(source_name, symbol, "events", "funding"):
            stored = store.read(source_name, symbol, "events", "funding", cfg.start, cfg.end)
            funding_events[symbol] = stored["funding_rate"]
            sources.append(store.metadata(source_name, symbol, "events", "funding") or {})
    return frames, funding_events, sources


def load_dataset(
    config: DataConfig,
    cleaning: CleaningConfig | None = None,
    seed: int = 42,
) -> Dataset:
    """Load, clean, align and audit market data as configured.

    Sources:
        * ``synthetic``: deterministic simulated perp market (offline, seeded).
        * ``ccxt``: download through CCXT, cached in the Parquet store.
        * ``parquet``: read previously downloaded data only (no network).

    Raises:
        DataQualityError: if ``config.quality.on_fail == "error"`` and the audit fails.
        DataSourceError: if the exchange cannot be reached or rejects a request.
    """
    cleaning = cleaning or CleaningConfig()
    extras: dict[str, dict[str, pd.Series]] = {}
    events: dict[str, pd.Series] | None = None
    sources: list[dict[str, Any]] = []

    if config.source == "synthetic":
        market = generate_synthetic_market(config.symbols, config.timeframe, config.synthetic, seed)
        raw_frames = market.ohlcv
        funding_bars = market.funding if config.include_funding else {}
        extras["basis"] = market.basis
        frames, reports = _clean_all(raw_frames, config.timeframe, cleaning)
    else:
        store = ParquetStore(config.storage_dir)
        raw_frames, events, sources = _load_exchange(
            config, store, use_network=config.source == "ccxt"
        )
        frames, reports = _clean_all(raw_frames, config.timeframe, cleaning)
        funding_bars = {
            sym: aggregate_funding_to_bars(ev, frames[sym].index, config.timeframe)
            for sym, ev in events.items()
            if sym in frames
        }

    data = MarketData.from_frames(
        frames,
        config.timeframe,
        funding={s: f for s, f in funding_bars.items() if s in frames} or None,
        extras={k: {s: v for s, v in d.items() if s in frames} for k, d in extras.items()},
    )
    if config.start or config.end:
        data = data.loc(config.start, config.end)
    logger.info(
        "Loaded %d symbols x %d bars (%s, %s -> %s)",
        len(data.symbols),
        data.n_bars,
        config.timeframe,
        data.index[0].date(),
        data.index[-1].date(),
    )

    quality = assess_quality(data, config.quality, events if config.include_funding else None)
    if config.quality.on_fail != "ignore" and quality.table["flags"].ne("").any():
        (logger.info if quality.passed else logger.warning)(quality.one_line())
    enforce_quality(quality, config.quality.on_fail)
    return Dataset(data, reports, quality, sources)


def load_market_data(
    config: DataConfig,
    cleaning: CleaningConfig | None = None,
    seed: int = 42,
) -> tuple[MarketData, list[CleaningReport]]:
    """Load, clean and align market data: :func:`load_dataset` without the audit trail."""
    dataset = load_dataset(config, cleaning, seed)
    return dataset.data, dataset.cleaning
