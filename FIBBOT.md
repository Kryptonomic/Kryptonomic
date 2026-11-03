# fibbot user guide

fibbot is a bot that **paper trades** Fibonacci-retracement setups on the largest crypto coins by
market cap. It tests many versions of the strategy on past prices, keeps only those that pass
strict statistical tests, then paper trades the survivors on new prices. It calls a strategy
"reliably profitable" only after it has also passed a second gate on its paper results.

A Claude "brain" runs the research: it decides what to test next, reads the results and keeps a
notebook. Plain code, not Claude, enforces every risk limit, budget and gate.

## What it is, and what it is not

**It is:**

- A research lab for one family of strategies: buying pullbacks to Fibonacci levels (and selling
  rallies, if you allow shorts).
- A paper trader. Every account is simulated, starting at $10,000 per strategy.
- Honest by design. A locked validation period, a random-entry baseline (on the research data,
  on the validation period and again on paper), a correction for the number of looks at the
  locked data, and a forward paper period all stand between an idea and the label "reliably
  profitable".

**It is not:**

- **A live trading bot.** It has no exchange keys and contains no code that sends orders.
  Nothing it does can move real money.
- **Financial advice.** Its results, Claude's notes and the reports are research output, not
  recommendations. Past performance, on paper or in a backtest, does not predict future results.
- A guarantee. A strategy can pass every test and still lose money later. See
  [Limitations](#limitations).

## How it works

```text
  Binance candles (1h / 4h / 1d)          CoinGecko market caps
            |                                    |
            v                                    v
   +-----------------+               +----------------------+
   |  candle cache   |<--------------|  coin universe       |
   +-----------------+               |  top 12, no stables  |
            |                        +----------------------+
            v
   +-----------------+
   |  Fib engine     |   the SAME code runs backtests and paper trading
   +-----------------+
            |
            v
   +------------------------------+   RESEARCH WINDOW: all history except the last 12 months
   | backtest + walk-forward      |
   | + random-entry baseline      |
   | + robustness (one-step moves)|
   +------------------------------+
            |  only the best ideas
            v
   +------------------------------+   VALIDATION WINDOW: the last 12 months, kept locked
   | validation gate (pass/fail)  |   (pinned once a strategy is on paper)
   | must beat random entries too |
   +------------------------------+
            |  passed
            v
   +------------------------------+   new candles only, $10,000 simulated per strategy
   | paper trading (up to 4)      |
   +------------------------------+
            |  60+ days and 40+ closed trades
            v
   +------------------------------+
   | graduation gate (pass/fail)  |   a report, never a switch to live trading
   +------------------------------+

  Claude brain: weekly research, daily review, a look at each new signal
    - reads the lab status and its own notebook
    - works only through tools: run_backtest, run_optimization, run_baseline,
      run_robustness, validate_strategy, promote_to_paper, retire_strategy, ...
    - has a budget per cycle, cannot change the gates or risk limits, and cannot trade
```

The steps in plain words:

1. **Data.** Closed candles from Binance spot (OKX fills recent gaps). Saved on disk, then
   updated a few bars at a time.
2. **Universe.** The 12 largest coins by market cap that trade against USDT on Binance and have at
   least a year of history. Stablecoins and wrapped tokens (WBTC, stETH, ...) are left out. The list
   is re-ranked every 7 days.
3. **Backtest.** The engine replays history bar by bar. It never looks ahead: a decision at the
   close of a bar uses only that bar and earlier ones, and orders fill on later bars.
4. **Walk-forward.** The research data is cut into 4 time slices. Settings are chosen using the
   early slices only, then scored on the next slice the optimizer has not seen. This shows whether
   "tuned on the past" still works "on the future". The slices are unseen by the optimizer but not
   by the researcher (Claude sees every slice's backtest results first), so walk-forward measures
   how stable the tuning is, not independent evidence. Only the locked validation period and paper
   trades are unseen.
5. **Random baseline.** The same trades, with the same stop distance, entry cost and targets, but
   entered at random times on the same coins and window. A second set of random entries is drawn
   only at times the strategy's trend / ADX filter allowed. This tests entry selection (the setup
   plus its filter) against random timing; judge on the smaller of the two edges. If the Fibonacci
   entries do not beat random entries, there is no real edge, and the bot will not promote it.
6. **Robustness.** Every setting is moved one step up and one step down (up to 48 neighbours).
   Neighbours that trade exactly like the original are flagged as duplicates and left out of the
   score. A real edge survives small changes; a lucky setting does not.
7. **Validation gate.** A fixed pass/fail test on the locked last 12 months. That data is never
   used for tuning. The coins are chosen as they stood before the period started (by trading
   volume), not by today's ranking. The strategy must also beat random entries on that period by a
   significant margin. The bar rises with every look at the locked period (the "deflated
   Sharpe"), and each strategy gets one look per period. Once a strategy is on paper the period is
   pinned: it no longer slides forward over the data paper trades on.
8. **Paper trading.** A strategy that passes starts paper trading on the next new candle. Every
   hour the bot re-runs the same engine over the new candles, so paper equals a backtest of the
   same period. History is never rewritten: a closed trade's prices and R are final, a trade that
   a code or data change would back-date is refused, and an open trade that the replay stops
   reproducing for 3 updates in a row, or whose coin's candles fall more than 3 days behind, is
   closed at the latest close, with costs (exit reason `void` or `data_gap`).
9. **Graduation gate.** Every hour each paper strategy is checked against a second fixed test that
   uses its paper results only. Passing it takes at least 60 days with candles and 40 closed
   paper trades, and in practice much longer.

The gates are ordinary code with fixed thresholds (see [Graduation criteria](#graduation-criteria)).
Claude can propose, test and promote ideas, but it cannot pass a gate by arguing.

## The Fibonacci strategy family in plain words

Every strategy is the same idea with different settings. Here it is for a long (buy) trade;
shorts are the mirror image.

**1. Find the swing.** The bot finds swings with a "ZigZag". A swing low is confirmed once price
has climbed a set distance above it (by default 3 x ATR, the average bar size). That low is the
**anchor**. The highest high since the anchor is the **extreme**. Anchor to extreme is the **leg**.
It must be big enough to matter (by default at least 4 ATR).

**2. Draw the levels.** Retracement levels measure how far price has pulled back from the extreme
toward the anchor:

- 0.382 = 38.2% of the way back (a shallow dip)
- 0.5 = halfway back
- **0.618 to 0.65 = the "golden pocket"**, the most popular Fibonacci zone
- **0.618 to 0.786 = "OTE"** (optimal trade entry), a wider and deeper zone
- 0.886 = a very deep pullback
- 1.0 = all the way back to the anchor (the setup is broken)

**3. Check the trend.** Most presets only buy pullbacks while price is above its 200-period EMA at
the moment the high was made. Some add ADX (trend strength) or require a higher low than the last
swing.

**4. Enter.** There are two styles:

- **Limit entry:** a resting buy order inside the zone (top, middle or bottom). It fills if price
  trades down to it.
- **Confirm entry:** wait until price dips into the zone, then wait for a sign of buyers within a
  few bars before buying at the next open. The sign is one of: a close back above the zone
  ("reclaim"), a bullish engulfing candle ("engulf"), or RSI turning up from below 40 ("rsi").

**5. Stop and targets are Fibonacci ratios too.**

- **Stop:** a deeper level than the zone, for example 0.886 or 1.0 (the anchor), plus a small ATR
  buffer.
- **Target 1:** usually 1.0 as an extension, which is the old high (the extreme). By default half
  the position is closed there and the stop moves to breakeven.
- **Target 2:** an extension beyond the old high, usually 1.618. Some presets trail the rest
  with an ATR stop instead.
- **Time stop:** a trade still open after a set number of bars (default 120) is closed.
- A setup is only taken if the planned reward is at least 1.5 times the risk. That ratio is
  before costs. `min_stop_cost_mult` (0-20, off by default) is the lever against stops so tight
  that costs eat them, which mostly matters on 1h: 3 means the stop must be at least three
  round-trip costs away (about 0.75% of the price with the default costs in limit mode). On 6
  coins it cut the losses of `golden_pocket_1h` from -0.097 R to -0.048 R per trade; it stayed
  negative.

**A worked example.** BTC rises from an anchor at 60,000 to an extreme at 70,000. The leg is
10,000. ATR is 800.

| Item | Ratio | Price |
|---|---|---|
| Golden pocket top | 0.618 | 63,820 |
| Golden pocket bottom | 0.65 | 63,500 |
| Limit buy (zone middle) | 0.634 | 63,660 |
| Stop | 0.886 minus 0.25 ATR | 60,940 |
| Target 1 (old high) | 1.0 extension | 70,000 |
| Target 2 | 1.618 extension | 76,180 |

Risk is 2,720 per coin. Target 1 pays 2.3 times that and target 2 pays 4.6 times. With half the
position at each target, the planned reward is about 3.5 times the risk, so the trade passes the
1.5 minimum.

**Results are counted in R.** R is profit divided by the amount risked. +1R means you made what you
risked; -1R is a full stop-out. With the default 0.5% risk per trade, 1R is 0.5% of the account.
The profit factor is gross winning R divided by gross losing R, so it is above 1 exactly when the
total R is positive. R statistics count every trade equally, while the account under-sizes trades
whose stop is very tight (a position may not exceed 25% of the account, so any stop closer than 2%
risks less than 0.5%). In the measured data this makes the R-based gates more conservative.

**How the simulator fills orders.** It is cautious on purpose:

- A candle only shows open, high, low and close, not their order. When one candle touches both
  the stop and a target, the stop is assumed to have come first.
- A candle that makes a new high is never treated as the dip that sets up a confirm entry.
- A resting limit entry fills only if price trades 0.02% through it (`costs.limit_through`). A
  bare touch does not fill: the setups that just tag a level and bounce are the best trades, and
  a queued order often misses them. Targets still fill on a touch.
- On the candle a trade opened on, only the stop is checked during the candle (its high may have
  come before the fill). If that candle closes beyond a target, the target is paid at its price.
- A fill at or beyond the stop (the candle opened through it) is stopped out at once, and its R is
  measured against the planned risk of the order.
- Each side of a trade pays 0.1% in fees. Market and stop orders also lose 0.05% to slippage.
- No leverage: all open positions together may not exceed the account's equity.

**Built-in presets** (`python -m fibbot presets`):

| Preset | Timeframe | Idea |
|---|---|---|
| `golden_pocket_4h` | 4h, long | Zone 0.618-0.65, limit at the middle, stop 0.886 + 0.25 ATR, half at 1.0, rest at 1.618, EMA200 trend |
| `golden_pocket_1h` | 1h, long | Same on 1h |
| `golden_pocket_both_4h` | 4h, long + short | Same, both directions |
| `ote_confirm_1h` | 1h, long | Zone 0.618-0.786, confirm (reclaim), stop 1.0 + 0.1 ATR |
| `shallow_trend_4h` | 4h, long | Strong trends only (EMA stack + ADX 20+), zone 0.382-0.5, limit at 0.382, stop 0.786 |
| `deep_786_4h` | 4h, long | No trend filter, zone 0.786-0.886, stop 1.0 + 0.3 ATR, targets 0.618 (most of the way back up) and 1.0 (the old high) |
| `golden_pocket_1d` | 1d, long | Zone 0.5-0.65, limit at the middle, stop 0.886 |
| `ext_runner_4h` | 4h, long | Golden pocket, 30% at 1.0, trail the rest 2.5 ATR, target 2.618 |

Claude and the optimizer explore around these by changing the 28 settings (zone, entry style,
stop, targets, trend filter, swing size, minimum stop distance and so on).

## Quick start

You need Python 3.10 or newer.

```bash
pip install -r requirements.txt          # the Claude SDK, pinned to exact versions (only the brain needs it)
export ANTHROPIC_API_KEY=sk-ant-...      # optional: without it the bot runs without the brain
python -m fibbot data                    # first download of candles: about 5-15 minutes
python -m fibbot sweep                   # test the 8 presets (no Claude needed), report only
python -m fibbot run                     # run around the clock (stop with Ctrl+C)
```

`requirements.txt` pins the exact versions of the Claude SDK and everything it depends on, so an
unattended install never picks up a new release. Upgrading is deliberate: bump the pins, run
`pip install -r requirements.txt`, then run the tests (`python -m pytest tests`).

What `run` does every hour, just after the candle closes:

1. Refresh the coin list if it is older than 7 days, then download the new candles.
2. Update every paper strategy: new signals, fills and exits.
3. Ask Claude about each new signal (by default this is only recorded, see
   [Configuration](#configuration)).
4. Check every paper strategy against the graduation gate. A strategy is retired automatically if
   its paper drawdown goes past 22.5% (1.5 x the 15% limit). Without Claude, a strategy whose
   status is "failing" is retired too, so a loser cannot block the automatic research.
5. Run Claude's daily review and weekly research cycle when they are due. While no strategy is
   on paper there is nothing to review, so research runs daily instead, except while the weekly
   validation cap is used up (it could not validate anything). Without Claude, an empty roster
   triggers the automatic preset sweep instead, at most once a week.
6. Rewrite the report at `.fibbot/REPORT.md` and, about once a day, compact the database (old
   experiment details and old system-log entries; nothing a gate reads).

If a step fails, the error goes into the journal and the log, and the remaining steps still run.
The same failure of the same step is written to the journal at most once a day (the log still
gets it every time). A Claude cycle that fails or crashes is not retried on every pass: the next
try comes 6 hours after the failed one started, so a flaky API cannot run up the bill. A cycle
also stops after `brain.max_cycle_minutes` (60): it gets one last reply to write its notebook
entry and summary.

**When candles stop arriving.** One coin failing to download is only a warning. When every
download fails, an open paper position's candles are more than 3 bars old, or no coin has a
recent bar on a timeframe, the data step fails on every pass (so `run --once` exits 1 and an
Actions run is marked failed), the stall is journaled once when it starts and once when it ends,
and the report shows a **STALE DATA** banner. The review cycle waits for fresh candles; research
continues on the cached history. Days without candles do not count toward the 60 graduation days.

Read the results with `python -m fibbot status`, or open `.fibbot/REPORT.md`.

## Commands

Run `python -m fibbot <command>`. Two options work with every command and go before it:
`--config PATH` (a config file) and `--home DIR` (the state folder, default `.fibbot`).
Run `python -m fibbot <command> --help` for the exact options of each command.

| Command | What it does |
|---|---|
| `universe [--refresh]` | Show the coin list. `--refresh` re-ranks it now. |
| `data [--tf 1h 4h 1d] [--symbols BTCUSDT ...]` | Download or update candles. |
| `presets` | List the built-in strategies and their settings. |
| `backtest --preset NAME` | Backtest one strategy. Options: `--window research` (default), `validation` or `all`; `--symbols ...`; `--tf 1h` to try another timeframe; `--trades N` to list the last N trades. |
| `optimize --preset NAME --grid JSON` | Walk-forward search over a grid, for example `--grid '{"entry_zone_hi": [0.65, 0.705, 0.786]}'`. The grid can also be a JSON file. |
| `baseline --preset NAME [--runs N]` | Compare the strategy with random entries (10 runs by default). |
| `robustness --preset NAME` | Move every setting one step both ways and re-test (duplicates of the original are flagged). |
| `validate --preset NAME` | Run the validation gate on the locked last 12 months. One look per strategy per validation period. |
| `promote --preset NAME --rationale TEXT [--name LABEL]` | Put a validated strategy on paper (at most 4 at a time). It needs a passed validation and a `baseline` run showing an edge over random entries. |
| `retire ID --reason TEXT` | Take a strategy off paper. Its trades stay on record, and retirement is final: a retired strategy can never be promoted again (promote a variant with a changed setting instead). |
| `epoch [--roll]` | Show the pinned validation period. `--roll` moves it forward by `research.validation_months` (journaled; the new period overlaps data your paper strategies traded on). |
| `sweep [--promote] [--presets NAME ...]` | Test the presets (by default every preset whose timeframe is in `data.timeframes`) with a baseline. A preset is eligible with enough trades, a positive mean R, at least half of the folds positive and an edge over random entries. The top 3 eligible ones are validated, and with `--promote` the passing ones go on paper. |
| `tick` | Update candles and the paper accounts once, now (exit code 1 if anything failed, including a download). |
| `brain research`, `brain review`, `brain signals` | Run one Claude research cycle, review cycle or signal review now. |
| `run [--once] [--no-brain]` | The full loop. Ctrl+C stops it after the current step. `--once` does one pass and exits (exit code 1 if a step failed). `--no-brain` never calls Claude. |
| `status` | A summary in the terminal. |
| `report` | Write `.fibbot/REPORT.md`. |

Anywhere `--preset NAME` works, `--spec` works too. It takes your own settings as JSON or a JSON
file. To change a preset, name it and list only the changes:

```bash
python -m fibbot backtest --spec '{"preset": "golden_pocket_4h", "entry_zone_hi": 0.705}'
```

`backtest`, `optimize`, `baseline`, `robustness`, `validate`, `sweep` and `status` also take
`--json` to print machine-readable output.

Do not use `--window validation` or `--window all` while you are still exploring ideas. Each
look at the locked period makes its verdict less trustworthy. Such a run prints a warning, counts
as a trial and as a look at that strategy's held-out period (raising the bar for every later
validation), is stored with source `cli-holdout` with its numbers hidden from Claude, and is
counted as "CLI holdout peeks" in `status` and REPORT.md. Use `validate` once you have a finished
candidate.

## Graduation criteria

**First gate: validation** (on the locked last 12 months, before paper trading):

| Check | Default |
|---|---|
| Closed trades | at least 30 |
| Profit factor (gross winning R / gross losing R) | at least 1.2 |
| Average trade (expectancy) | at least +0.05 R |
| Deflated Sharpe (corrected for every look at the locked period, trades clustered by week) | at least 0.90 |
| Maximum drawdown of the simulated account (marked to market) | at most 20% |
| Share of traded coins that made money | at least 50% |
| Average trade minus that of random entries on the same period (50 runs) | at least 0 R |
| z score of that edge | at least 1.65 |

How the validation stays honest:

- The coins are the top coins by trading volume in the 90 days before the period started, not
  today's list, which holds coins because they rallied during that year. The pool is today's
  list, every coin the bot has seen in a list, the static fallback list and any cached candles
  (coins you download with `data --symbols` join it).
- Each strategy gets one look per validation period. Validating the same strategy again is
  refused while the period overlaps the earlier one (the earlier verdict stands, and the
  automatic sweep reuses it). A changed setting is a new strategy and a new look.
- For promotion, every validation of the strategy in the last `validation_months` must have
  passed, and its research-period baseline must beat random entries.
- Claude's cycles and the automatic sweep share `research.max_validations_per_week` (3). Your own
  `validate` commands are not capped, but count.
- Once the first strategy is on paper, the validation period is pinned: later data is forward
  data only and never becomes validation data by sliding. `epoch --roll` moves it on purpose.

**Second gate: graduation** (on paper trades only). Every check must pass:

| Check | Default |
|---|---|
| Days on paper | at least 60 |
| Closed paper trades | at least 40 |
| Profit factor | at least 1.3 |
| 95% anytime-valid confidence interval of the average trade, trades clustered by week | lower end above 0 R |
| Maximum drawdown of the paper account | at most 15% |
| Share of calendar months that made money (months with trades) | at least 60% |
| Deflated Sharpe (corrected for every strategy ever put on paper) | at least 0.95 |
| Average paper trade minus that of random entries over the paper period | at least 0 R |
| z score of that edge | at least 1.65 |
| Paper average trade vs. validation average trade | at most 0.25 R worse |

"Days on paper" counts only days the strategy had candles for. The interval is "anytime-valid":
the gate is re-checked every hour, and an ordinary 95% interval re-tested that often would let
about 1 in 10 strategies with no edge graduate within a year by luck. With this interval it is
about 2.5% or less, at the cost of time.

A paper strategy always shows one of five statuses:

- **collecting**: not enough days or trades yet.
- **promising**: enough data and a positive average trade, but not every check passes yet. Give
  it time.
- **lagging**: enough data, but the paper average trade is zero or negative. It becomes failing
  if that is still so after three times the minimum days (180).
- **failing**: a drawdown above 22.5%; or clearly losing (with at least 20 trades: profit factor
  below 1 and even the optimistic end of the interval is negative); or, once mature, even the
  optimistic end of the interval is below the validation average minus 0.25 R; or still not
  profitable after 180 days.
- **graduated**: every check passed. This is a statement about paper results. Nothing switches
  to live trading. What you do with it is your decision.

Expect this to take a year or more. A strategy whose true edge is +0.3 R per trade, with about two
trades every three days, graduates within a year about 55-65% of the time; +0.25 R about 40%; +0.2 R
about half the time within two years. Most ideas fail along the way, which is the point.

## Cost of the Claude API

The bot itself is free. Claude calls are billed by Anthropic per token (roughly, per word read and
written). The default model is Claude Opus 5.5. At the time of writing it costs $4 per million
input tokens and $20 per million output tokens. Repeated context read from the prompt cache costs
$0.20 per million, and the bot caches heavily. Current prices are on the
[Anthropic pricing page](https://platform.claude.com/docs/en/about-claude/pricing).

Rough estimates with the default settings:

| Claude call | How often | Rough cost each | Rough cost per month |
|---|---|---|---|
| Research cycle (up to 60 model turns, effort "high") | weekly; daily while no strategy is on paper | $2 - $10 | $10 - $40 |
| Review cycle (at most 20 turns, effort "medium", no optimizer) | daily | $0.50 - $3 | $15 - $90 |
| Signal review (one short question per new setup) | per signal, at most 40 calls a day | $0.02 - $0.07 | $1 - $20 typical |

The worst case of one cycle is its turn cap times `brain.max_tokens` (64,000 output tokens per
reply, thinking included) times the output price, plus the cache reads: about 60 x 64,000 x $20 /
million = $77 for a research cycle that used every turn to the limit, and 20 x 64,000 x $20 /
million = $26 for a review cycle. Real cycles stop far below that; the review cycle's cost is
bounded by its turn cap and effort. Each cycle's token usage is in its journal entry (with the
turn cap, effort, resumes after API errors and whether it wrapped up), and the kv entry
`brain_usage` tallies tokens per month.

While no strategy is on paper, research runs daily and there are no reviews or signals, so
expect roughly $2 to $10 a day until the first strategy is promoted.

Once strategies are on paper, expect something like **$30 to $150 a month** with the defaults.
These are order-of-magnitude guesses. The real cost depends on how many signals your strategies
produce, how long Claude's cycles run and how much it thinks. The journal records the tokens each cycle used, so after a
week you can work out your own number. You can also check usage in the Claude Console.

Ways to spend less (see [Configuration](#configuration)):

- `"brain": {"effort": "medium"}`: Claude thinks less on each step.
- `"brain": {"review_every_hours": 72}`: review every 3 days instead of daily. This also slows
  research to every 3 days while no strategy is on paper.
- `"brain": {"signal_review": "off"}`: no per-signal calls at all.
- `"brain": {"max_iterations": 30}`: shorter research cycles; `"review_max_iterations": 10` for
  reviews.
- `"brain": {"max_monthly_output_tokens": 5000000}`: a hard monthly cap on output tokens. Once it
  is reached, cycles and signal reviews stop until the next UTC month and the automatic preset
  sweep stands in.
- `"brain": {"model": "claude-sonnet-5-5"}`: about half the price per token ($2 / $10 per
  million), with some loss of research quality.
- Run with no API key, or with `run --no-brain`. That costs nothing, and research is limited to
  the automatic preset sweep.

## Running 24/7

The bot needs to wake up once an hour. If it misses some hours, nothing is lost: the next update
replays the missed candles and records the same trades it would have recorded on time.

### On your own computer

```bash
python -m fibbot run
```

Keep the terminal open, for example inside `tmux` or `screen`, or start it in the background:

```bash
nohup python -m fibbot run > fibbot.out 2>&1 &
```

The computer must stay on and awake. A laptop that sleeps pauses the bot.

### With Docker

```bash
docker build -t fibbot .
docker run -d --name fibbot --init --stop-timeout 300 --restart unless-stopped \
  -e ANTHROPIC_API_KEY \
  -v fibbot-data:/data \
  fibbot
```

- All state lives in the `fibbot-data` volume, mounted at `/data`. It survives restarts and
  rebuilds.
- `-e ANTHROPIC_API_KEY` passes the key from your shell. Leave it out to run without the brain.
- `--init` forwards `docker stop`'s signal to the bot, which finishes its current step (a Claude
  cycle wraps up at its next turn) before exiting. Docker's default grace period is 10 seconds,
  after which it kills the bot mid-step and a paid cycle in progress is lost; `--stop-timeout 300`
  (compose: `stop_grace_period: 5m`) gives it time. A kill never corrupts the state.
- Your own config: add `-v "$PWD/fibbot.json:/app/fibbot.json:ro"`. Do not set `home` in it: the
  image uses `FIBBOT_HOME=/data`, which takes precedence over a config file's `home` anyway.
- Follow the log: `docker logs -f fibbot`.
- Run other commands inside it: `docker exec fibbot python -m fibbot status`.
- Read the report: `docker exec fibbot cat /data/REPORT.md`.
- The image runs as a non-root user (uid 10001). A named volume, as above, just works. If you
  bind-mount a host folder instead, make it writable for that user, or add
  `--user "$(id -u):$(id -g)"`.

### On GitHub Actions (no computer of your own needed)

The workflow `.github/workflows/fibbot-paper.yml` runs one bot pass every hour, at 7 minutes past.

1. Push this repository to GitHub. The workflow file must be on the default branch.
2. In the repository, open **Settings > Secrets and variables > Actions**.
3. On the **Variables** tab, add `FIBBOT_ENABLED` with the value `true`. Without it the job is
   skipped. Set it to `false` to pause the bot.
4. On the **Secrets** tab, add `ANTHROPIC_API_KEY`. This is optional; without it the bot runs
   without the brain.
5. Optional config: `fibbot.json` is git-ignored, so commit your settings under another name, for
   example `config/fibbot.json`. Then add a variable `FIBBOT_CONFIG` with that path. Do not set
   `home` in that file.
6. Open the **Actions** tab, choose **fibbot paper trading**, and press **Run workflow** to
   start the first run now.

Where things are kept:

- **State** (database, notebook, report) is on the branch `fibbot-state`. Each run replaces that
  branch with a single fresh commit, so it never builds up history. Open `REPORT.md` on that
  branch to read the latest report. The report is also shown on each run's summary page.
- **Candles** are kept in the Actions cache, not in git. If the cache is evicted, the next run
  downloads them again.
- To start over from nothing, delete the `fibbot-state` branch.
- To study the state on your own computer:

  ```bash
  git fetch origin fibbot-state
  mkdir -p .fibbot && git archive FETCH_HEAD | tar -x -C .fibbot
  python -m fibbot status
  ```

Things to know:

- **Public repository:** the `fibbot-state` branch is public too, including Claude's notebook and
  your paper trades. Your API key is never written to it. Use a private repository if you want
  privacy.
- **Private repository:** Actions minutes are limited by your GitHub plan, and a run every hour
  uses many of them. To run less often, edit the `cron` line, for example
  `"7 */4 * * *"` for every 4 hours. Missed hours are caught up, as explained above.
- The first run downloads years of candles and may start a research cycle, so it can take up to
  an hour or more. Later runs usually take a few minutes. Runs never overlap.
- If one step of a run fails (for example, every candle download failed), the run is marked
  failed and GitHub may email you. The state and candles are still saved, also after a timeout
  (the bot step stops after 150 minutes, inside the job's 180) or a manual cancel, and the next
  run carries on. The error is in the run log and in the journal.
- The checkout does not keep the GitHub token while the bot runs; only the restore and save steps
  get it.
- The save step compacts the database, warns above 50 MB and refuses to push (a failed step with
  an error message) when a state file is over 95 MB, because GitHub rejects files over 100 MB. The
  bot compacts old experiment details daily, so this should not happen for a long time.
- GitHub can delay scheduled runs at busy times. In public repositories it pauses schedules
  after 60 days without activity. Re-enable the workflow on the Actions tab if that happens.
- The save step force-pushes to `fibbot-state`. A rule that blocks force-pushes on all branches
  will make it fail.
- If candle downloads fail with HTTP 451 or 403 errors, the exchange is blocking the region of
  GitHub's servers. These (and 418 IP bans) are not retried, so each coin costs one quick refused
  request. Recent candles may still arrive from OKX, so a blocked runner can look healthy apart
  from "Binance klines failed ... trying OKX" warnings in the log. Use Docker on a machine
  elsewhere instead.
- If the CoinGecko ranking fails (for example rate-limited on shared runner IPs), the bot keeps
  the last ranking (or the static list) and tries again 6 hours later;
  `python -m fibbot universe --refresh` retries at once.

## State files

Everything the bot knows lives in one folder, `.fibbot/`. Which folder: `--home DIR` wins, then
the `FIBBOT_HOME` environment variable, then `"home"` in the config file, then `./.fibbot` (a
warning is logged when the file and the environment disagree):

| File | What it holds |
|---|---|
| `fibbot.sqlite` | The database: every experiment, the paper roster, paper trades, signals and Claude's reviews of them, equity curves, the journal and settings the bot remembers. |
| `candles/SYMBOL_TF.csv.gz` | Downloaded price history. Safe to delete; it is downloaded again. |
| `universe.json` | The current coin list and when it was ranked. |
| `exchange_symbols.json` | Cached list of Binance spot pairs. |
| `BRAIN_NOTEBOOK.md` | Claude's research notebook. It reads this at the start of every cycle. |
| `REPORT.md` | The readable report, rewritten every run. |
| `fibbot.log` | The run log (the same lines `run` prints on screen), rotated at 5 MB with 3 old files kept (`fibbot.log.1` .. `.3`). |

Back up the folder by copying it while the bot is stopped. To start completely fresh, stop the
bot and delete the folder.

## Configuration

The bot works with no config file. To change settings, create `fibbot.json` in the folder you run
the bot from. Alternatively, point `--config` or the `FIBBOT_CONFIG` environment variable at a
file. List only the settings you want to change; everything else keeps its default.
`fibbot.example.json` is a small example:

```json
{
  "universe": {"top_n": 10},
  "costs": {"fee_rate": 0.001, "slippage": 0.0005, "limit_through": 0.0002},
  "portfolio": {"risk_per_trade": 0.0025, "max_total_notional": 1.0},
  "research": {"max_validations_per_week": 3, "validation_universe": "point_in_time",
               "validation_gate": {"min_validation_edge_r": 0.0, "min_validation_edge_z": 1.65}},
  "brain": {"signal_review": "shadow", "effort": "medium", "max_cycle_minutes": 60}
}
```

The config is checked when it is loaded. An unknown key (a typo such as `"efort"`, with a "did you
mean" hint), a wrong type, or an out-of-range or unknown value stops the command with an error
that names the key (exit code 1). Values are case-sensitive and never normalised:
`signal_review` must be exactly `off`, `shadow` or `active`, and efforts `low`, `medium`, `high`,
`xhigh` or `max`.

A paper account's costs, portfolio rules and starting equity are frozen when the strategy is
promoted. Editing `costs.*`, `portfolio.*` or `paper.starting_equity` later only affects
strategies promoted afterwards, so a config change cannot re-price a running paper record.

The settings you are most likely to change (all defaults are in `fibbot/config.py`):

| Setting | Default | Meaning |
|---|---|---|
| `universe.top_n` | 12 | How many of the largest coins to trade. |
| `universe.extra_exclude` | `[]` | Coins to leave out, for example `["TRX"]`. |
| `data.history_start` | `"2020-01-01"` | Earliest candle to download. |
| `costs.fee_rate` / `costs.slippage` | 0.001 / 0.0005 | Fee per side, and slippage on market and stop orders. |
| `costs.limit_through` | 0.0002 | A limit entry fills only when price trades this fraction through it (0 = fill on touch; at most 0.05). |
| `portfolio.risk_per_trade` | 0.005 | Risk 0.5% of the account per trade (to the stop). |
| `portfolio.max_open_positions` | 6 | Most trades open at once. |
| `portfolio.max_total_risk` | 0.03 | All open trades together risk at most 3%. |
| `portfolio.max_notional_frac` | 0.25 | No position larger than 25% of the account (no leverage). |
| `portfolio.daily_loss_limit` | 0.02 | No new trades for the rest of the UTC day after a 2% realized loss. It rarely binds on 4h and is not a material protection. |
| `portfolio.max_total_notional` | 1.0 | All open positions together at most 1x the account: no leverage. |
| `paper.max_roster` | 4 | Strategies paper trading at once. |
| `paper.starting_equity` | 10000 | Starting balance of each paper account. |
| `research.validation_months` | 12 | Length of the locked validation period. |
| `research.max_validations_per_week` | 3 | Validations in the last 7 days across Claude's cycles and the automatic sweep (`null` disables). |
| `research.validation_universe` | `"point_in_time"` | Validation coins ranked by volume before the period, or `"current"` (today's list). |
| `research.validation_gate.min_validation_edge_r` / `min_validation_edge_z` | 0.0 / 1.65 | Validation edge over random entries, and its z score (`null` drops a check). |
| `graduation.min_edge_r` / `graduation.min_edge_z` | 0.0 / 1.65 | The same on the paper period. |
| `brain.enabled` | `true` | `false` never calls Claude. |
| `brain.model` | `"claude-opus-5-5"` | Claude model for every brain call. |
| `brain.effort` / `brain.signal_effort` | `"high"` / `"medium"` | How hard Claude thinks in cycles and in signal reviews: `low`, `medium`, `high`, `xhigh` or `max`. |
| `brain.signal_review` | `"shadow"` | `off`, `shadow` or `active` (see below). |
| `brain.max_signal_reviews_per_day` | 40 | Cap on per-signal Claude calls per rolling 24 hours. Failed calls (refusal, cut-off or malformed reply, API error) count too, and a signal is given up after 2 failed attempts. |
| `brain.research_every_hours` / `brain.review_every_hours` | 168 / 24 | Weekly research, daily review. |
| `brain.max_tokens` | 64000 | Output tokens per reply in cycles, thinking included. Replies are streamed, so values above 21,000 are fine; 64,000 is recommended. |
| `brain.max_iterations` / `brain.review_max_iterations` | 60 / 20 | Model turns per research / review cycle. |
| `brain.review_effort` | `"medium"` | How hard Claude thinks in review cycles. |
| `brain.max_cycle_minutes` | 60 | Wall-clock limit of one cycle; it then gets one wrap-up reply (`0` or `null` = no limit). |
| `brain.max_retries` | 8 | SDK retries per request (it honours retry-after). |
| `brain.signal_max_tokens` | 16000 | Output tokens per signal review, thinking included (at most 21,000). |
| `brain.max_monthly_output_tokens` | `null` (off) | Monthly cap on output tokens. When reached, brain calls stop until next month and the automatic sweep stands in. |

**Signal review modes:**

- `off`: Claude does not look at single signals.
- `shadow` (default): Claude says "take" or "skip" for each new signal, but the paper account
  trades every signal anyway. The report then compares the trades Claude would have skipped with
  the ones it liked. That tells you whether its judgment adds anything.
- `active`: signals Claude says to skip are not traded on paper. A late "skip" never removes a
  trade that already filled.

Setups whose order already filled (or closed) are not reviewed: a late skip could not remove them
and would be scored with hindsight. The track record Claude sees for a signal (paper trades and
its own past reviews) is as it stood at the signal's candle.

**Claude's journal.** Every cycle writes a journal entry with its summary, the tools it used and
its token usage. A cycle cut off by its turn cap, its output limit or `brain.max_cycle_minutes`
ends with one extra reply that writes the notebook entry and the summary; the entry shows the cap,
effort, resumes and whether it wrapped up.

The gate thresholds (`research.validation_gate` and `graduation`) can be changed too. Loosening
them makes "reliably profitable" mean less.

## Limitations

- **Survivorship bias.** The research period uses today's top coins with at least a year of
  history. Coins that crashed out of the top ranks are missing, and today's top coins rallied
  into it, so long backtests look better than reality would have been. Judge research results on
  their edge over random entries, not on the raw average trade. The validation period uses the
  coins as they stood before it started (by trading volume; the pool is the coins the bot has seen,
  cached coins and the static list) and must beat random entries. Paper trading uses the ranking
  as it changes, which removes the bias from that point on.
- **Candles hide what happened inside them.** The simulator only sees open, high, low and close.
  It resolves every unclear candle against the trade (stop before target, no entry on the candle
  that made the high). That is cautious but still an estimate. A limit entry needs price to trade
  0.02% through it, which models "a touch may not fill you"; on 6 coins this costs
  `golden_pocket_4h` about 15% of its total R. Targets still fill on a touch.
- **Tight stops and costs.** On 1h, stops are often so close that fees and slippage take a large
  part of every R. `min_stop_cost_mult` filters such setups; it has not made the 1h presets
  profitable.
- **Fees and slippage are assumptions.** 0.1% per side and 0.05% slippage suit liquid coins on a
  normal day. In a fast market, real slippage can be many times larger. The bid-ask spread is not
  modelled beyond that.
- **Shorts are simplified.** Short trades are simulated as if you could sell short at spot prices
  with no borrowing or funding costs. In reality shorting needs margin or futures, and those
  cost money.
- **Markets change.** A pattern that worked in 2021-2025 can stop working. The validation window
  and the paper period reduce this risk; they do not remove it.
- **Many ideas, some lucky.** The deflated Sharpe and the gates correct for the number of ideas
  tried, but no test removes false positives completely.
- **Paper is not live.** No real orders means no rejected orders, no exchange outages, no partial
  fills, no delays, and no real money on the line.
- **Data can be wrong or missing.** It comes from public exchange endpoints that can fail or be
  revised.
- **Claude can be wrong.** It runs the research but cannot pass a gate by itself. Its notes are
  hypotheses, not advice.
- **Past performance does not predict future results.** This is a research tool, not financial
  advice.

## Research results

### First research run, 2026-11-03: nothing passed

**Bottom line:** fibbot has not found a reliably profitable Fibonacci strategy yet, so nothing is
paper trading. One candidate had strong research numbers, but it lost money in the locked
validation year.

**Setup**
- Coins: the top 12 by market cap on 2026-11-03 (BTC, ETH, BNB, XRP, SOL, TRX, ZEC, DOGE,
  LINK, ADA, XLM, BCH).
- Candles: 1h, 4h and 1d, 2020-01-01 to 2026-11-03.
- Costs: the defaults (0.1% fee and 0.05% slippage per side).
- Research window: 2020-01-01 to 2025-11-02.
- Validation year: 2025-11-02 to 2026-11-03, locked. It is scored on the coins that were top 12 on
  2025-11-02, so XLM and BCH are replaced by AVAX and SUI.

**How the research was done**
- Four research agents did the brain's job through the CLI on the research window only. Each took
  one family:
  1. shallow, trend-aligned pullbacks;
  2. exit design;
  3. confirmation entries, regime filters and shorts;
  4. costs and leg size on 1h.
- The store recorded 355 trials, so the multiple-testing count is honest.
- Three judges (overfitting, regime and recency, mechanism and costs) re-ran every candidate. Every
  reported number reproduced exactly.
- Only one candidate got the judges' votes, so only one of the three weekly validation looks was
  used.

**Candidates.** "Edge" is the avg R above random entries with the same side, risk and exits, on
both random draws (uniform / same regime), with z-scores: 50 random runs for the first two
rows, 10 for the rest. The timeframe is 4h unless the idea says otherwise.

| spec | idea | research trades | avg R | edge (z) | 2022 / 2024 / 2025 avg R | judges |
|---|---|---|---|---|---|---|
| `aa47d5f033` | `shallow_trend_4h` + `require_structure` + `entry_at: mid` + ADX 25 + `ema_slope` | 566 | +0.295 | +0.25 (3.1) / +0.23 (2.8) | +0.04 / +0.08 / +0.19 | 3 of 3: validate |
| `72e8b44f2e` | same rule with `ema_stack` | 488 | +0.311 | +0.26 (3.0) / +0.22 (2.5) | +0.14 / +0.13 / +0.15 | 0 of 3: shares 93-96% of trades with the first, so it is the same bet |
| `213150f1f1` | `golden_pocket_4h`, no trend filter, ADX 25, structure | 529 | +0.268 | +0.22 (2.2) / +0.19 (1.9) | -0.00 / +0.06 / +0.00 | 0 of 3: decayed; walk-forward out-of-sample +0.01 |
| `8fe000ae40` | `golden_pocket_1h`, `min_stop_cost_mult` 10, `min_leg_atr` 6 (1h) | 1064 | +0.155 | +0.17 (2.1) / +0.15 (1.8) | -0.12 / +0.25 / +0.04 | 0 of 3: cost-fragile, 71% of R in 2020-21 |
| `22cd5281c9` | `ext_runner_4h` on 1d, trail 4 ATR (1d) | 208 | +0.763 | +0.31 (1.4) / +0.37 (1.7) | -0.92 / +0.89 / +0.16 | 0 of 3: mostly market drift plus a few big winners |

**Validation of `aa47d5f033` (one look, failed)**
- 82 trades, avg -0.22 R, profit factor 0.65, max drawdown 6.9%.
- Edge over random entries -0.20 R (z -1.38).
- 4 of 12 coins and 1 of 4 quarters positive.
- It failed 6 of the 8 checks: profit factor, expectancy, deflated Sharpe, symbol win fraction and
  both edge checks.

The other two variants of the same idea were deliberately not validated. Picking them after seeing
this result would be fitting to the validation year.

**What was learned** (details and numbers are in the notebook)
- **The presets as shipped do not beat random entries.** For example, `golden_pocket_4h` beats
  random by 0.02 R and `shallow_trend_4h` by 0.09 R (z 1.5). Most of their raw profit is the
  2020-2021 bull market.
- **Timing edge came from filters.** A higher-low anchor (`require_structure`), a strong trend
  (ADX at least 25) and a slightly deeper limit inside the shallow zone added the edge. It held in
  research, not in the validation year.
- **Exits did not create an edge on 4h.** Looser trails and bigger targets raise random entries
  just as much.
- **Losing ideas:** deep zones (0.786 and deeper), confirmation entries on 4h, and shorts. Shorts
  only made money in 2022, and both directions together is always below long-only.
- **1h needs a cost filter and stays fragile.** It only works with `min_stop_cost_mult` around 10,
  which really selects high-volatility legs.
- **One validation year has low power.** At about 80 trades a year, an edge needs a research z of
  about 4 to have a good chance of passing the year. That is why ideas with more trades are worth
  more.

**Disclosure.** Before this store existed, the code review ran all 8 presets on the same
validation year in throwaway stores. All 8 failed; the best was `golden_pocket_4h` at about
+0.005 R. The research agents and judges were not told this. The lead, who picked the validation
process, knew it. The store's trial and look counts do not include those throwaway runs.

**What happens next**
- With an empty roster the hourly loop keeps researching weekly: the brain if an API key is set,
  otherwise the deterministic autopilot.
- Validation stays capped at 3 looks a week and one look per spec per period.
- The validation year slides forward as new candles arrive, until the first promotion pins it.
- Forward paper trading is the only fully fresh test once something passes.
- The brain notebook from this run is in
  [`research/2026-11-03-brain-notebook.md`](research/2026-11-03-brain-notebook.md). For a new
  deployment, copy it to `$FIBBOT_HOME/BRAIN_NOTEBOOK.md` so the brain does not repeat the dead
  ends.
- The notebook's open ideas:
  - structure + ADX 25 on the 1h cost-filtered base;
  - the shallow structure rule on 1h;
  - the same rule without a trend filter;
  - a market-wide (BTC) regime gate, which needs a code change.
