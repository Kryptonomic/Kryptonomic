# Brain notebook

## 2026-11-03 First research run (research window 2020-01-01..2025-11-02)
Four explorer families (trend_pullbacks, exits_runners, confirm_regime, lower_tf_costs), about 500 backtest-equivalents. Three judges (overfitting, regime, mechanism) re-ran every candidate's backtest and baseline; all numbers reproduced. holdout_peeks 0; trial_stats.n about 355 after exploring.

### Reading the tools
- run_baseline copies each trade's stop, target R distances, trail, breakeven and time stop: baseline_avg_r is what the exits earn on random entries, edge_r is what entry timing adds.
- Random-entry drift: 4h shallow specs about +0.04 R; golden_pocket_4h longs +0.05 to +0.08; 1h gp with msc 10 about -0.01; 1d longs +0.26 to +0.46 (more with wider runners).
- A filter regime can carry drift: random entries earn +0.111 R in ema_stack + ADX 25, +0.042 in ema_slope + ADX 25. Use min(edge_r, edge_regime_r).
- 10-run regime draws can be lenient: aa47d5f033 regime edge 0.253 (z 3.03) at 10 runs, 0.228 (z 2.78) at 50. A baseline costs 1 unit whatever --runs: use 30-50 for finalists.
- Edge z uses max(strategy clustered SE, baseline run sd): at about 208 trades (1d) SE is about 0.21, so a 0.30 R edge gives z about 1.4 even at beats_frac 1.0.
- Raw-R robustness hides edge decay: 22cd5281c9 had 32/32 neighbours positive, but baselined neighbours had edges of 0.05-0.22. Baseline 2-3 neighbours before trusting a plateau.
- Robustness costs 33-37 units per spec: finalists only. Pass --trades 5000 on first runs to post-process trades.
- The baseline draws over the whole window, so there is no per-year edge: read per-year raw R against the overall drift.
- min_rr 1.25-1.75 and min_stop_cost_mult 0.5 never bound on 4h/1d specs here (duplicate neighbours).
- Holdout power: one-year z is about research z x 0.41 (1/sqrt 5.83 years). Research z 3 -> about 1.2, z 2 -> about 0.85, vs the 1.65 gate. A good pass chance needs research z about 4, or 400+ trades a year at the same edge.

### Validation choice
- Validated: aa47d5f033 = {preset shallow_trend_4h, require_structure true, entry_at mid, adx_min 25, trend_filter ema_slope}. 3/3 judges voted validate, mean score 6.7/10.
- aa47d5f033: n 566 (about 95 a year), avg R +0.2952, PF 1.623, DD 9.2%.
- aa47d5f033 edge: 10-run 0.2538 (z 3.04) / regime 0.253 (z 3.03); 50-run 0.2547 (z 3.11) / 0.2281 (z 2.78).
- aa47d5f033 years 2020-2025: +0.372/+0.469/+0.044/+0.438/+0.079/+0.188. Folds +0.474/+0.195/+0.347/+0.114. 12/12 half-years positive. 10/12 coins positive (BTC -0.11, ADA -0.04).
- aa47d5f033 robustness: 31/31 distinct neighbours positive, median +0.265, worst +0.124 (swing_mode pct). ADX 20/30 +0.226/+0.265; leg 3/5 +0.257/+0.298.
- aa47d5f033 costs: median stop 6.4% of price, drag about 0.047 R; 2x costs +0.248, 3x +0.201. Without its top 5% of trades +0.198.
- aa47d5f033 2022-2025: n 331, +0.20 R, week-clustered t 2.04. 2024-2025: +0.124 (t 0.88), carried by about 5 coins.
- Why chosen: the only candidate with binding edge z near 2.8 on 50 runs, every year positive, an interior plateau point, and each ingredient carried edge alone.
- Risks: picked greedily from about 25 specs (ADX 25 and ema_slope from scans). WF OOS on the sister base +0.039/+0.433/+0.066. Expected holdout z 1.0-1.3 even if real: pass odds about 20-35%.
- The 99-110 legs ema_slope adds over ema_stack average +0.064 R (2024+ -0.074); the shared core carries the recent edge (2022+ +0.242, t 2.9).
- Not validated: 72e8b44f2e (same rule, ema_stack). n 488, +0.311, PF 1.67, every year positive; regime edge 0.201 (z 2.21), 50-run 0.219 (z 2.45). 93-96% of its trades are aa47d5f033's: the same bet.
- Not validated: 213150f1f1 {golden_pocket_4h, trend_filter none, adx_min 25, require_structure true}. n 529, +0.268, PF 1.42, binding z 2.14 (50 runs); 2022/2024/2025 -0.004/+0.059/+0.000; WF OOS +0.0097, combo changed every fold; found after about 45 specs.
- Not validated: 8fe000ae40 {golden_pocket_1h, min_stop_cost_mult 10, min_leg_atr 6}. n 1064, +0.155, PF 1.24, regime z 1.81; 2022 -0.122, 2025 +0.040; 71% of R in 2020-2021; without top 5% of trades -0.012.
- Not validated: 22cd5281c9 {ext_runner_4h, timeframe 1d, trail_atr 4}. n 208, +0.763 but random +0.453; edge z 1.44 / 1.74; 2022 -0.915; about 36 trades a year.

### What adds timing edge (4h longs)
- shallow_trend_4h preset d25afc4741: n 903, +0.116, PF 1.25, edge 0.085 (z 1.45) / 0.070 (z 1.21); 2025 -0.015. No edge as it stands.
- On that preset: entry_at mid (0.441) +0.178, edge 0.151 (z 2.23); require_structure +0.170, edge 0.152 (z 2.38), 12/12 coins.
- Structure + mid 058b419aac: n 647, +0.233, edge 0.180 (z 2.29) / 0.187 (z 2.47), 2025 -0.007.
- On structure + mid: ADX 15/20/25/30 = +0.228/+0.253/+0.310/+0.291; min_leg_atr 4/5/6 = +0.259/+0.297/+0.256. WF OOS over that grid +0.174 (n 299).
- With ADX 25 + structure + mid the trend filter barely matters: ema +0.262 (n 610), ema_slope +0.295 (566), ema_stack +0.311 (488).
- In-sample best cell ADX 25 + leg 5 92253dd0a7: +0.329 but 2022 -0.039. Not chosen.
- Gains do not stack: leg 6 + mid 34eb365eea 2025 -0.062; structure + mid + leg 6 0a788b0d7f 2022 -0.109. ADX 25 alone 33ebe4fa1e +0.147, 2025 -0.029.
- Shallow entry depth: 0.382 +0.196; 0.441 +0.295 to +0.311; 0.5 +0.268 to +0.298 on fewer trades.
- golden_pocket_4h edge_r: base 0.021; +ADX 25 0.036 (regime 0.008: ADX alone is regime, not timing); +structure 0.064; ADX 25 + structure + EMA200 0.138; same with trend_filter none 0.216.
- On gp4h ADX 25 + structure, stricter trend filters lower the binding edge: ema_slope 0.108 (z 1.02), ema_stack 0.101 (z 0.90), ema 0.132 (z 1.28).
- Depth 0.618-0.70 behaves alike under the gp4h gate (+0.262 to +0.288); OTE 0.618-0.786 mid 89a36ccc7e +0.237, binding z 1.36.
- swing_mode pct is the worst neighbour on every robustness run (+0.07 to +0.15, half the folds positive): the edge needs ATR-scaled pivots.

### 1h and 1d
- 1h: min_stop_cost_mult is the lever. golden_pocket_1h at msc 0/6/10/16: avg R -0.070/+0.038/+0.124/+0.180, n 4337/2935/1480/554, edge_r 0.049/0.098/0.140 at msc 0/6/10 (0.166 with leg 6).
- 1h msc 10 is a big-%-leg (high volatility) selector, not just cost relief: wider stops are worse (stop_atr_buffer 0.5 +0.089, 1.0 +0.019, vs 0.25 +0.155); min_leg_atr 12 without the floor -0.099.
- 1h msc at leg 6: 8 +0.056, 10 +0.155, 12 +0.150, 14 +0.12. msc 12 (leg 6) and msc 16 turn 2025 negative (-0.082, -0.291).
- At msc 10 on 1h BTC barely trades (26 trades); the R comes from DOGE, SOL, BCH, LINK, XRP.
- 1d entry 0.618-0.65, stop 0.886 + 0.25 ATR beats random under all 7 exit designs tried (edge 0.156-0.439), but z stays 1.1-1.9: about 210 trades caps it.

### Dead ends (do not retest)
- Depth 0.786+: deep_786_4h c343ee318f -0.105, edge -0.074; confirm -0.154; gated -0.06 to -0.13; 0.786 entry under the 213150f1f1 gate (4896c272b0) edge -0.019.
- Confirm entries (reclaim, engulf, rsi) on 4h: about 0.1 R worse than limit under the same gate; ungated +0.005 to +0.042.
- Shorts lose in every year except 2022: gp4h short 52571fa124 -0.227 (0/12 coins); with ADX 25 + structure -0.132. direction both is always below long-only.
- golden_pocket_4h preset 8450eeebe8: +0.075, edge 0.021; 2022 -0.216, 2025 -0.154. Bull artifact.
- 4h exits cannot create edge: 10 designs gave raw +0.045 to +0.208, edge 0.012-0.066 (z <= 0.76); looser trails lift the random baseline equally.
- No effect: breakeven, max_hold 240, trailing on shallow. Worse: all out at tp1, 40-bar time stop, tp1_ext 0.8 on 1d.
- min_stop_cost_mult 10-20 on 4h raises raw R, not edge (z <= 1.24).
- golden_pocket_1d preset 94a858f2fa: +0.319 but random +0.303, edge 0.015; 2022 -0.905. 1d variants with structure, confirm, leg 6 or the shallow zone: edge <= 0.036, or unstable on too few trades.
- Larger legs on 4h: gp4h leg 8 -0.031; WF OOS of leg x msc grid -0.074.
- Plain 1h gp aaa3bbf571 -0.070; ote_confirm_1h 97f60ed19c -0.083 (+0.058 with msc 10 + leg 6, 2025 -0.087).
- Shallow zone 0.5-0.618 with stop 0.886 (959b0bafed): +0.200, 2022 -0.067, leans on BNB.

### Common weaknesses
- Every long candidate makes 60-71% of its R in 2020-2021; 2022, 2024H1 and 2025H1 are weak in every 4h long variant.
- No exit or short variant rescues 2022: long losses there are full -1R stops.
- The universe is today's top coins (survivorship); ADA and BTC are negative in several candidates.
- The strategy space has no market-wide (BTC) regime gate; testing one needs a code change.

### Open hypotheses / next steps
- If aa47d5f033 fails validation, the 4h structure + ADX 25 pullback thesis failed for this period: do not validate 72e8b44f2e or 213150f1f1 (small variations, 54-96% shared legs).
- Holdout power binds: prefer ideas that keep about 0.2 R of edge at 200+ trades a year over higher avg R on fewer trades.
- Open: require_structure + ADX 25 on the 1h msc 10 base (structure added edge on 4h; 1h gives about 180 trades a year).
- Open: the shallow structure + mid rule on 1h with msc 10.
- Open: shallow structure + mid + ADX 25 with trend_filter none (on gp4h, dropping the EMA raised the binding edge 0.132 -> 0.189).
- Check list_experiments(params=...) before running the open ideas: some may exist as robustness neighbours.

## Validation results
_pending_
