# Copy-trade plan: his entries + our 2x-initials rule

Target wallet: `ffQUA7xYw32XT9Kjgak5pekN3hSLAgUyhSdYkUMGdmd`

## The idea

| His strengths (keep) | Our strength (add) |
|---|---|
| Finds coins early (27% of his coins reach 2x from his entry) | **Never let a 2x round-trip:** take initials at 2x if he hasn't already |
| Adds size only to coins that are working (his conviction signal) | Stop adding once initials are out (house money only) |
| Cuts losers within minutes (median hold 4 min) | Hard daily loss limit and form-based sizing |
| Scales out on pumps (NEARCAT: sold 45% at 2x, then chunks from 4x to 18x) | |

## Bot rules

### Entry
1. **Copy every buy he makes**, first buys and adds, at **1.0x his SOL amount**.
   Start the first week at **0.5x** while you check your fills.
2. **Max slippage 5%.** Skip the buy if your fill would be more than 5% above his.
3. **High priority fee / Jito tip.** You need to land within 1 to 3 seconds of him.
   Backtest: every +1% on entry costs about 1 SOL per 80 SOL traded.
4. **No per-coin cap below what you can afford.** His biggest adds are his biggest
   winners (7UrgFT: 13.6 SOL in, best coin in the backtest). A 1-2 SOL cap turned
   the plan from about break-even to -2 to -3 SOL.
   Set the cap to the most you are OK losing on one coin.
5. **Once initials are out, ignore his further adds on that coin.**
   If he fully exits and later re-buys, treat it as a new position.

### Exit
6. **Copy his sells proportionally.** If he sells 30% of his bag, sell 30% of yours.
   This includes his limit-order fills. If your bot can't see those (they're executed
   by a keeper, not signed by him), turn on "copy on token transfer out" if the bot has it.
7. **Take initials at 2x.** When price hits 2x of your average entry, sell just enough
   to get back the SOL you put in, *minus anything his sells already returned to you*.
   If his sells already returned your initial, this does nothing. The earlier
   "sell half at 2x" version double-sold on NEARCAT and left only 27% riding.
8. **No trailing stop.** Every trailing stop we tested (20-50%) cut big runners early.
   On NEARCAT a 20% trail sold everything at 3.7x; the coin went to 19.8x and his
   scale-out made +1.65 SOL vs +0.45 with the trail.

### Risk
9. **Daily loss limit: -1 SOL** (scale with size). Stop copying for the rest of the day.
10. **Size by his form.** If his wallet is up over the last 7 days, trade 1x.
    If it's down, trade 0.5x or pause. Copying him in a losing week loses too.
11. **Re-run the backtest weekly** (`python3 backtest/backtest.py`) and adjust.

## Backtest results

Data: his last 1,000 transactions (Sep 13-27, 2026), 130 coins with minute-level
prices. Includes 1% bot fee per swap and 0.003 SOL priority fee per tx.

| Your entry vs his | Him | This plan (1x) | Last 5 days (his hot streak) |
|---|---|---|---|
| +1% | -0.2 SOL | **+0.1 SOL** on 71 SOL deployed | **+5.4 SOL** |
| +2% | -0.2 SOL | -1.0 SOL | +5.0 SOL |

- **Fast entry matters:** your extra entry cost is estimated from his own
  back-to-back buys. Landing 1-4 seconds after him cost a median +0.4 to +0.9%.
- **The plan tracks his form:** week 1 (he lost): plan -4.6 SOL. Week 2 (he won): plan +3.6 SOL.
- **2x size, no cap** at +1%: +1.7 SOL over two weeks, but -8.6 in week 1 and +8.2 in week 2.

### With a zero-fee bot (pump fees only)

Pump's fees are already in his fill prices, so the only extra costs are slippage and priority fees.

| Size | Setup | Entry +1% | Entry +2% |
|---|---|---|---|
| 1.0x | pure copy | -3.2 | -4.7 |
| 1.0x | **plan (copy + 2x initials)** | **+1.4** | **+0.3** |
| 1.15x | pure copy | -3.5 | -5.2 |
| 1.15x | **plan** | **+1.8** | **+0.5** |
| 2.1x | pure copy | -5.2 | -8.3 |
| 2.1x | **plan** | **+4.5** | **+2.2** |

Him over the same period: -0.2 SOL. Plan at 1.0x/+1%, Sep 23-27: +6.2 SOL.
Run with `--fee 0` to reproduce.

### Other variants tested (for the record)

| Variant | 2 weeks @+1% | NEARCAT @+2% (him: +1.65) |
|---|---|---|
| Pure 1:1 copy | -4.8 | +1.49 |
| **This plan** | **+0.1** | **+1.17** |
| His sells until 2x, then half at 2x + 20% trail | +0.8 | +0.46 |
| Ignore his sells; half at 2x + trail + 40% stop | -4.6 | ~+0.5 |
| Flat 0.5 SOL per coin, 2x initials | -2.1 (at 0%) | n/a |
| Laddered sells (15-25% every +30-50%) + trail | -0.2 to -1.0 | +0.4 to +0.8 |

The plan is roughly even with the best trailing-stop version over two weeks, but keeps
the upside on runners like NEARCAT, which is where his money is made.

## Caveats
- Two weeks of history; results follow his form.
- Minute candles; real fills on fast wicks will be somewhat worse.
- 34 of 164 coins had no usable price data and are excluded.
