# Kryptonomic

Copy-trading research for Solana memecoin wallets.

- `strategy/COPY_TRADE_PLAN.md`: the plan (his entries + our 2x-initials rule), bot rules, backtest results
- `strategy/bot-config.json`: the same rules as settings to enter in a copy-trade bot
- `backtest/backtest.py`: replays the plan against a wallet's real on-chain trades

```
python3 backtest/backtest.py                    # default wallet, last 1000 txs
python3 backtest/backtest.py --slip 0.01 --mult 1.0
python3 backtest/backtest.py --wallet <address>
```

The copy-trade backtest is standard library only. Set `HELIUS_API_KEY` to fetch through Helius instead of the rate-limited public RPC. The first run downloads data into `.cache/` (a few minutes; public RPC
and GeckoTerminal are rate-limited), and later runs reuse it. Delete `.cache/` to refresh.

## fibbot: Fibonacci-retracement paper trading

`fibbot/` is a separate research bot that **paper trades** Fibonacci-retracement setups on the
largest crypto coins, with a Claude "brain" that directs the research. It contains no live-trading
code and never places real orders; nothing it produces is financial advice. See
[FIBBOT.md](FIBBOT.md) for the user guide and `fibbot/DESIGN.md` for the module contract.

```
pip install -r requirements.txt     # only the Claude brain needs it (pinned anthropic SDK)
python -m fibbot data               # download candles (first run: minutes)
python -m fibbot sweep              # test the presets without Claude
python -m fibbot run                # hourly paper trading and research
```

The core of fibbot is standard library only; `requirements.txt` is needed only for the brain.
To run it unattended, use the `Dockerfile` or the GitHub Actions workflow
`.github/workflows/fibbot-paper.yml` (both described in FIBBOT.md).

Status: the first research run (2026-11-03) found no strategy that passes the validation year,
so nothing is paper trading yet. See [Research results](FIBBOT.md#research-results).
