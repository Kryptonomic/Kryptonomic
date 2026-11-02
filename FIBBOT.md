# fibbot user guide (draft)

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
- Honest by design. A locked validation period, a random-entry baseline, a correction for the
  number of ideas tried, and a forward paper period all stand between an idea and the label
  "reliably profitable".

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
   | + robustness (small nudges)  |
   +------------------------------+
            |  only the best ideas
            v
   +------------------------------+   VALIDATION WINDOW: the last 12 months, kept locked
   | validation gate (pass/fail)  |
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
   early slices only, then scored on the next slice they have not seen. This shows whether
   "tuned on the past" still works "on the future".
5. **Random baseline.** The same trades, with the same stop distance and targets, but entered at
   random times. If the Fibonacci entries do not beat random entries, there is no real edge.
6. **Robustness.** Each setting is nudged one step up and one step down. A real edge survives small
   changes; a lucky setting does not.
7. **Validation gate.** A fixed pass/fail test on the locked last 12 months. That data is never
   used for tuning. The bar rises with the number of ideas tried so far (the "deflated Sharpe").
8. **Paper trading.** A strategy that passes starts paper trading on the next new candle. Every
   hour the bot re-runs the same engine over the new candles. Past paper trades never change.
9. **Graduation gate.** Every hour each paper strategy is checked against a second fixed test that
   uses its paper results only. Passing it takes at least 60 days and 40 closed paper trades.

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
- A setup is only taken if the planned reward is at least 1.5 times the risk.

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

**How the simulator fills orders.** It is cautious on purpose:

- A candle only shows open, high, low and close, not their order. When one candle touches both
  the stop and a target, the stop is assumed to have come first.
- A candle that makes a new high is never treated as the dip that sets up a confirm entry.
- A target is never paid on the candle the trade opened on.
- Each side of a trade pays 0.1% in fees. Market and stop orders also lose 0.05% to slippage.

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

Claude and the optimizer explore around these by changing the 27 settings (zone, entry style,
stop, targets, trend filter, swing size and so on).

## Quick start

You need Python 3.10 or newer.

```bash
pip install -r requirements.txt          # installs the Claude SDK (only the brain needs it)
export ANTHROPIC_API_KEY=sk-ant-...      # optional: without it the bot runs without the brain
python -m fibbot data                    # first download of candles: about 5-15 minutes
python -m fibbot sweep                   # test the 8 presets (no Claude needed), report only
python -m fibbot run                     # run around the clock (stop with Ctrl+C)
```

What `run` does every hour, just after the candle closes:

1. Refresh the coin list if it is older than 7 days, then download the new candles.
2. Update every paper strategy: new signals, fills and exits.
3. Ask Claude about each new signal (by default this is only recorded, see
   [Configuration](#configuration)).
4. Check every paper strategy against the graduation gate. A strategy is retired automatically if
   its paper drawdown goes past 22.5% (1.5 x the 15% limit).
5. Run Claude's daily review and weekly research cycle when they are due. While no strategy is
   on paper there is nothing to review, so research runs daily instead. Without Claude, an empty
   roster triggers the automatic preset sweep instead, at most once a week.
6. Rewrite the report at `.fibbot/REPORT.md`.

If a step fails, the error goes into the journal and the log, and the remaining steps still run.
A Claude cycle that fails or crashes is not retried on every pass: the next try comes 6 hours
after the failed one started, so a flaky API cannot run up the bill.

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
| `robustness --preset NAME` | Nudge each setting one step and re-test. |
| `validate --preset NAME` | Run the validation gate on the locked last 12 months. |
| `promote --preset NAME --rationale TEXT [--name LABEL]` | Put a validated strategy on paper (at most 4 at a time). |
| `retire ID --reason TEXT` | Take a strategy off paper. Its trades stay on record. |
| `sweep [--promote] [--presets NAME ...]` | Test the presets (by default every preset whose timeframe is in `data.timeframes`) with a baseline. A preset is eligible with enough trades, a positive mean R, at least half of the folds positive and an edge over random entries. The top 3 eligible ones are validated, and with `--promote` the passing ones go on paper. |
| `tick` | Update candles and the paper accounts once, now. |
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

Do not use `--window validation` while you are still exploring ideas. Each look at the locked
period makes its verdict less trustworthy. Use `validate` once you have a finished candidate.

## Graduation criteria

**First gate: validation** (on the locked last 12 months, before paper trading):

| Check | Default |
|---|---|
| Closed trades | at least 30 |
| Profit factor (gross wins / gross losses) | at least 1.2 |
| Average trade (expectancy) | at least +0.05 R |
| Deflated Sharpe (corrected for all ideas tried) | at least 0.90 |
| Maximum drawdown of the simulated account | at most 20% |
| Share of traded coins that made money | at least 50% |

**Second gate: graduation** (on paper trades only). Every check must pass:

| Check | Default |
|---|---|
| Days on paper | at least 60 |
| Closed paper trades | at least 40 |
| Profit factor | at least 1.3 |
| 95% confidence interval of the average trade (bootstrap) | lower end above 0 R |
| Maximum drawdown of the paper account | at most 15% |
| Share of calendar months that made money (months with trades) | at least 60% |
| Deflated Sharpe (corrected for every strategy ever put on paper) | at least 0.95 |
| Paper average trade vs. validation average trade | at most 0.25 R worse |

A paper strategy always shows one of four statuses:

- **collecting**: not enough days or trades yet.
- **promising**: enough data, but not every check passes yet.
- **failing**: clearly losing (with at least 20 trades: profit factor below 1 and even the
  optimistic end of the confidence interval is negative), or drawdown above 22.5%.
- **graduated**: every check passed. This is a statement about paper results. Nothing switches
  to live trading. What you do with it is your decision.

Expect this to take months. Most ideas fail along the way, which is the point.

## Cost of the Claude API

The bot itself is free. Claude calls are billed by Anthropic per token (roughly, per word read and
written). The default model is Claude Opus 5.5. At the time of writing it costs $4 per million
input tokens and $20 per million output tokens. Repeated context read from the prompt cache costs
$0.20 per million, and the bot caches heavily. Current prices are on the
[Anthropic pricing page](https://platform.claude.com/docs/en/about-claude/pricing).

Rough estimates with the default settings:

| Claude call | How often | Rough cost each | Rough cost per month |
|---|---|---|---|
| Research cycle (up to 60 tool steps, effort "high") | weekly; daily while no strategy is on paper | $2 - $10 | $10 - $40 |
| Review cycle (smaller budget, no optimizer) | daily | $0.50 - $3 | $15 - $90 |
| Signal review (one short question per new setup) | per signal, at most 40 a day | $0.02 - $0.07 | $1 - $20 typical |

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
- `"brain": {"max_iterations": 30}`: shorter research cycles.
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
docker run -d --name fibbot --init --restart unless-stopped \
  -e ANTHROPIC_API_KEY \
  -v fibbot-data:/data \
  fibbot
```

- All state lives in the `fibbot-data` volume, mounted at `/data`. It survives restarts and
  rebuilds.
- `-e ANTHROPIC_API_KEY` passes the key from your shell. Leave it out to run without the brain.
- `--init` lets `docker stop` shut the bot down cleanly.
- Your own config: add `-v "$PWD/fibbot.json:/app/fibbot.json:ro"`.
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
- If one step of a run fails (for example, the exchange is briefly unreachable), the run is
  marked failed and GitHub may email you. The state and candles are still saved, and the next
  run carries on. The error is in the run log and in the journal.
- GitHub can delay scheduled runs at busy times. In public repositories it pauses schedules
  after 60 days without activity. Re-enable the workflow on the Actions tab if that happens.
- The save step force-pushes to `fibbot-state`. A rule that blocks force-pushes on all branches
  will make it fail.
- If candle downloads fail with HTTP 451 or 403 errors, the exchange is blocking the region of
  GitHub's servers. Use Docker on a machine elsewhere instead.

## State files

Everything the bot knows lives in one folder, `.fibbot/` (change it with `--home` or the
`FIBBOT_HOME` environment variable):

| File | What it holds |
|---|---|
| `fibbot.sqlite` | The database: every experiment, the paper roster, paper trades, signals and Claude's reviews of them, equity curves, the journal and settings the bot remembers. |
| `candles/SYMBOL_TF.csv.gz` | Downloaded price history. Safe to delete; it is downloaded again. |
| `universe.json` | The current coin list and when it was ranked. |
| `exchange_symbols.json` | Cached list of Binance spot pairs. |
| `BRAIN_NOTEBOOK.md` | Claude's research notebook. It reads this at the start of every cycle. |
| `REPORT.md` | The readable report, rewritten every run. |
| `fibbot.log` | The run log (the same lines `run` prints on screen). |

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
  "portfolio": {"risk_per_trade": 0.0025},
  "brain": {"signal_review": "shadow", "effort": "medium"}
}
```

The settings you are most likely to change (all defaults are in `fibbot/config.py`):

| Setting | Default | Meaning |
|---|---|---|
| `universe.top_n` | 12 | How many of the largest coins to trade. |
| `universe.extra_exclude` | `[]` | Coins to leave out, for example `["TRX"]`. |
| `data.history_start` | `"2020-01-01"` | Earliest candle to download. |
| `costs.fee_rate` / `costs.slippage` | 0.001 / 0.0005 | Fee per side, and slippage on market and stop orders. |
| `portfolio.risk_per_trade` | 0.005 | Risk 0.5% of the account per trade (to the stop). |
| `portfolio.max_open_positions` | 6 | Most trades open at once. |
| `portfolio.max_total_risk` | 0.03 | All open trades together risk at most 3%. |
| `portfolio.max_notional_frac` | 0.25 | No position larger than 25% of the account (no leverage). |
| `portfolio.daily_loss_limit` | 0.02 | No new trades for the rest of the UTC day after a 2% realized loss. |
| `paper.max_roster` | 4 | Strategies paper trading at once. |
| `paper.starting_equity` | 10000 | Starting balance of each paper account. |
| `research.validation_months` | 12 | Length of the locked validation period. |
| `brain.enabled` | `true` | `false` never calls Claude. |
| `brain.model` | `"claude-opus-5-5"` | Claude model for every brain call. |
| `brain.effort` / `brain.signal_effort` | `"high"` / `"medium"` | How hard Claude thinks in cycles and in signal reviews: `low`, `medium`, `high`, `xhigh` or `max`. |
| `brain.signal_review` | `"shadow"` | `off`, `shadow` or `active` (see below). |
| `brain.max_signal_reviews_per_day` | 40 | Cap on per-signal calls. |
| `brain.research_every_hours` / `brain.review_every_hours` | 168 / 24 | Weekly research, daily review. |

**Signal review modes:**

- `off`: Claude does not look at single signals.
- `shadow` (default): Claude says "take" or "skip" for each new signal, but the paper account
  trades every signal anyway. The report then compares the trades Claude would have skipped with
  the ones it liked. That tells you whether its judgment adds anything.
- `active`: signals Claude says to skip are not traded on paper. A late "skip" never removes a
  trade that already filled.

The gate thresholds (`research.validation_gate` and `graduation`) can be changed too. Loosening
them makes "reliably profitable" mean less.

## Limitations

- **Survivorship bias.** The coin list is today's top coins with at least a year of history. Coins
  that crashed out of the top ranks are missing from the backtests, so backtests look better than
  reality would have been. Paper trading uses the ranking as it changes, which removes this bias
  from that point on.
- **Candles hide what happened inside them.** The simulator only sees open, high, low and close.
  It resolves every unclear candle against the trade (stop before target, no entry on the candle
  that made the high). That is cautious but still an estimate. A limit order is counted as filled
  as soon as price touches it; on a real exchange a touch may not fill you.
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

_Pending first research run._
