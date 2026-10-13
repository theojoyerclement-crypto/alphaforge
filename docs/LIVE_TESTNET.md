# Running on an exchange testnet

`alphaforge paper CONFIG --broker ccxt-testnet` sends the paper trader's orders to an exchange
**sandbox** through CCXT. The strategies, portfolio construction and drawdown overlay are the
backtest's own code; only the `Broker` changes (`src/alphaforge/live/ccxt_broker.py`).

**Testnet only.** The broker refuses any exchange whose sandbox mode is off. There is no
production path in this repository.

## Setup

1. Create API keys on the Binance Futures testnet (https://testnet.binancefuture.com). They are
   separate from real Binance keys and hold test funds only.
2. Put them in environment variables, never in code or config files:

   ```bash
   export ALPHAFORGE_TESTNET_API_KEY=...        # PowerShell: $env:ALPHAFORGE_TESTNET_API_KEY="..."
   export ALPHAFORGE_TESTNET_API_SECRET=...
   ```

3. Run one step per closed bar (cron, Task Scheduler):

   ```bash
   alphaforge paper configs/binance_perps.yaml --broker ccxt-testnet
   ```

   Running it twice on the same bar does nothing the second time.

## What happens at each step

1. **Kill-switch gate.** If the switch tripped earlier, nothing is done until it is reset.
2. **Sync.** The broker reads the margin balance and the positions, then checks that the last
   closed bar is recent (stale data), that NAV is not too far below the session high (max loss),
   and that the venue's positions match the positions it expects (desync).
3. **Decide.** Same strategies and `PortfolioConstructor` as the backtest; the drawdown overlay
   watches the simulated full-exposure shadow book (overlay v2).
4. **Execute.** One market order per symbol that needs to move: amount rounded down to the
   venue's precision, skipped below its minimum notional, `reduceOnly` when it only reduces.

## Safety properties

| Risk | Control |
|---|---|
| Duplicate orders after a timeout or a crash | Deterministic client order id per (experiment, bar, symbol, side); the order is looked up before sending and after any network error |
| Trading on a production account | Refuses unless the exchange is in sandbox mode; keys only from environment variables |
| Positions drifting from what the strategy believes | Reconciliation at every sync; a gap above `--max-position-gap` (default 5% of NAV) trips the kill switch |
| Runaway losses | `--max-loss` (default 10% below the session high) trips the kill switch |
| Trading on old prices | `--max-stale-bars` (default 2) trips the kill switch |
| Transient network errors | Retries with exponential backoff, on network errors only |
| Restarting after a trip | The halted state is persisted; it must be reset by hand |

## Files

Under `paper/<experiment>/` (git-ignored):

| File | Content |
|---|---|
| `broker_state.json` | Expected positions, last processed bar, kill-switch state |
| `journal.jsonl` | One JSON object per event: `sync`, `reconciled`, `order_sent`, `order_filled`, `order_already_placed`, `order_found_after_error`, `order_rejected`, `kill_switch`, ... |
| `state.json` | Shadow book and drawdown-controller state (trader side) |
| `fills.csv`, `equity.csv` | Per-step fills and NAV |

Resetting the kill switch after investigating the journal:

```python
from alphaforge.live.ccxt_broker import CcxtTestnetBroker
broker = CcxtTestnetBroker.from_env("binance_perps", "1d", "paper/binance_perps/broker_state.json")
broker.reset_kill_switch(operator="your name")
```

## Known limits

- Market orders only, filled at whatever the testnet book offers; no execution algorithm.
- The shadow book used by the overlay is simulated with the backtest's cost model, starting
  from the configured `initial_capital`, not from the testnet balance.
- Testnet liquidity and prices are not the real market's; fills say nothing about real costs.
- The automated tests use an in-memory exchange (`tests/fake_exchange.py`); a run against the
  real testnet needs your keys and is not part of CI.
