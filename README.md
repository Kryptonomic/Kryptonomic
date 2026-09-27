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

Standard library only. The first run downloads data into `.cache/` (a few minutes; public RPC
and GeckoTerminal are rate-limited), and later runs reuse it. Delete `.cache/` to refresh.
