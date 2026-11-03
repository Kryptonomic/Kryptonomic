"""Pure performance statistics for trade lists and equity curves.

Conventions: a trade is a *win* when ret_pct > 0 and a *loss* otherwise (ret_pct <= 0). Ratios
whose denominator is zero (profit factor without losses, Sharpe of a constant series, ...) and
ratios beyond +/-CAP are reported as +/-CAP (or 0.0 when the numerator is 0), which keeps every
value finite and JSON-safe. Standard deviations are sample (n-1) estimates; skew and kurtosis are
population moments, as in the Bailey & Lopez de Prado formulas.

Trades of one strategy on many correlated coins are clustered in time: they open and close
together, so n trades carry less information than n independent draws. The inference helpers
take optional `groups` (one cluster label per value, e.g. the UTC entry week, see entry_week) and
correct for it: design_effect / effective_n (cluster-robust CR1 variance of the mean over the iid
one), bootstrap_mean_ci(groups=) (resamples whole clusters), deflated_sharpe(n_eff=) and
confidence_sequence(groups=).
"""
from __future__ import annotations

import math
import random
from collections import Counter
from datetime import datetime, timezone
from statistics import NormalDist, fmean, median, stdev
from typing import Iterable, Sequence

from .types import Trade

CAP = 99.0
DAY = 86400
WEEK = 7 * DAY
EULER_GAMMA = 0.5772156649
_NORMAL = NormalDist()


def _ratio(num: float, den: float) -> float:
    """num / den clipped to [-CAP, CAP]; den <= 0 -> +/-CAP by the sign of num (0.0 if num == 0)."""
    if den <= 0:
        return math.copysign(CAP, num) if num else 0.0
    return max(-CAP, min(CAP, num / den))


def _sharpe(xs: Sequence[float]) -> float:
    """Mean / sample stdev; 0.0 for fewer than 2 values."""
    return _ratio(fmean(xs), stdev(xs)) if len(xs) >= 2 else 0.0


def _quantile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an ascending, non-empty sequence."""
    pos = q * (len(sorted_xs) - 1)
    i = min(int(pos), len(sorted_xs) - 1)
    j = min(i + 1, len(sorted_xs) - 1)
    return sorted_xs[i] + (sorted_xs[j] - sorted_xs[i]) * (pos - i)


def trade_stats(trades: Iterable[Trade]) -> dict:
    """Win/loss, R-multiple and return statistics of a trade list (keys per DESIGN.md).

    Trades are ordered by entry_t (stable) for max_consec_losses; profit_factor = gross winning R /
    gross losing R (sum of positive r_multiple / |sum of negative r_multiple|), which matches
    risk-based sizing (every trade risks the same share of equity, so its P&L is proportional to
    R; a ret_pct-based factor would weigh trades by stop width instead). profit_factor > 1 exactly
    when total_r > 0. sharpe_per_trade = mean / sample stdev of r_multiple. Empty input -> all
    zeros.
    """
    ts = sorted(trades, key=lambda tr: tr.entry_t)
    n = len(ts)
    out = {"n": n, "wins": 0, "win_rate": 0.0, "avg_r": 0.0, "expectancy_r": 0.0, "median_r": 0.0,
           "avg_win_r": 0.0, "avg_loss_r": 0.0, "total_r": 0.0, "profit_factor": 0.0,
           "avg_ret_pct": 0.0, "avg_bars_held": 0.0, "max_consec_losses": 0, "long_n": 0,
           "short_n": 0, "exit_reasons": {}, "sharpe_per_trade": 0.0}
    if n == 0:
        return out
    rs = [tr.r_multiple for tr in ts]
    win_r = [tr.r_multiple for tr in ts if tr.ret_pct > 0]
    loss_r = [tr.r_multiple for tr in ts if tr.ret_pct <= 0]
    gross_win = math.fsum(r for r in rs if r > 0)
    gross_loss = -math.fsum(r for r in rs if r < 0)
    streak = max_streak = 0
    for tr in ts:
        streak = 0 if tr.ret_pct > 0 else streak + 1
        max_streak = max(max_streak, streak)
    avg_r = fmean(rs)
    out.update(
        wins=len(win_r), win_rate=len(win_r) / n, avg_r=avg_r, expectancy_r=avg_r,
        median_r=median(rs), avg_win_r=fmean(win_r) if win_r else 0.0,
        avg_loss_r=fmean(loss_r) if loss_r else 0.0, total_r=math.fsum(rs),
        profit_factor=_ratio(gross_win, gross_loss) if gross_win > 0 else 0.0,
        avg_ret_pct=fmean(tr.ret_pct for tr in ts), avg_bars_held=fmean(tr.bars_held for tr in ts),
        max_consec_losses=max_streak, long_n=sum(tr.side > 0 for tr in ts),
        short_n=sum(tr.side < 0 for tr in ts), exit_reasons=dict(Counter(tr.exit_reason for tr in ts)),
        sharpe_per_trade=_sharpe(rs))
    return out


def max_drawdown(values: Iterable[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction of the peak (0.0 if none)."""
    peak, mdd = -math.inf, 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def _sorted_curve(curve: Iterable[tuple[int, float]]) -> list[tuple[int, float]]:
    """Curve points ordered by time; equal times keep their input order (last one wins later)."""
    return sorted(curve, key=lambda p: p[0])


def daily_closes(curve: Iterable[tuple[int, float]]) -> list[float]:
    """Last equity value of each UTC day from the curve's first to last day, gaps forward-filled."""
    closes = {int(t) // DAY: v for t, v in _sorted_curve(curve)}
    if not closes:
        return []
    out, last = [], None
    for d in range(min(closes), max(closes) + 1):
        last = closes.get(d, last)
        out.append(last)
    return out


def equity_stats(curve: Iterable[tuple[int, float]], start_equity: float) -> dict:
    """Return/risk statistics of an equity curve [(t, equity), ...].

    Daily returns come from the daily closes (see daily_closes) with start_equity as the base of
    the first day; Sharpe/Sortino are annualized with sqrt(365) and CAGR with 365 days per year.
    max_drawdown uses every raw curve point (plus start_equity), so intraday troughs count.
    n_days = number of calendar days spanned by the curve.
    """
    pts = _sorted_curve(curve)
    out = {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0,
           "calmar": 0.0, "n_days": 0}
    if not pts or start_equity <= 0:
        return out
    closes = daily_closes(pts)
    series = [start_equity] + closes
    rets = [b / a - 1.0 if a > 0 else 0.0 for a, b in zip(series, series[1:])]
    ratio, n_days = closes[-1] / start_equity, len(closes)
    cagr = math.exp(min(709.0, math.log(ratio) * 365.0 / n_days)) - 1.0 if ratio > 0 else -1.0
    mdd = max_drawdown([start_equity] + [v for _, v in pts])
    ann = math.sqrt(365.0)
    if len(rets) >= 2:
        mean = fmean(rets)
        downside = math.sqrt(fmean(min(r, 0.0) ** 2 for r in rets))
        out.update(sharpe=_ratio(mean * ann, stdev(rets)), sortino=_ratio(mean * ann, downside))
    out.update(total_return=ratio - 1.0, cagr=cagr, max_drawdown=mdd, calmar=_ratio(cagr, mdd),
               n_days=n_days)
    return out


def monthly_returns(curve: Iterable[tuple[int, float]], start_equity: float) -> list[tuple[str, float]]:
    """[("YYYY-MM", return), ...] for every UTC calendar month from the curve's first to last month.

    A month's return is its last equity value over the previous month's (start_equity for the
    first month); months without points carry the previous value forward (return 0.0).
    """
    ends = {}
    for t, v in _sorted_curve(curve):
        dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
        ends[(dt.year, dt.month)] = v
    if not ends:
        return []
    (y, m), last_ym = min(ends), max(ends)
    out, prev = [], start_equity
    while (y, m) <= last_ym:
        cur = ends.get((y, m), prev)
        out.append((f"{y:04d}-{m:02d}", cur / prev - 1.0 if prev > 0 else 0.0))
        prev = cur
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def entry_week(t: int) -> int:
    """UTC week number of a timestamp (t // WEEK): the cluster label of a trade's entry time."""
    return int(t) // WEEK


def _labels(xs: Sequence[float], groups) -> list:
    gs = list(groups)
    if len(gs) != len(xs):
        raise ValueError(f"{len(gs)} group labels for {len(xs)} values")
    return gs


def design_effect(values: Iterable[float], groups: Iterable) -> float:
    """Cluster-robust (CR1) variance of the mean over the iid variance of the mean, clipped to >= 1.

    CR1 = G/(G-1) * sum_g (sum_{i in g} (x_i - mean))^2 / n^2 over the G distinct labels in
    `groups` (one per value); iid = s^2 / n. Singleton clusters give exactly 1. A single cluster
    holding n >= 2 non-constant values gives n (one independent observation). Fewer than 2 values
    or a constant series -> 1.0. ValueError if the lengths differ.
    """
    xs = list(values)
    gs = _labels(xs, groups)
    n = len(xs)
    if n < 2:
        return 1.0
    mu = fmean(xs)
    ss = math.fsum((x - mu) ** 2 for x in xs)
    if ss <= 0:
        return 1.0
    sums: dict = {}
    for x, g in zip(xs, gs):
        sums[g] = sums.get(g, 0.0) + (x - mu)
    k = len(sums)
    if k < 2:
        return float(n)
    cr = k / (k - 1) * math.fsum(e * e for e in sums.values())
    return max(1.0, cr * (n - 1) / (n * ss))


def effective_n(values: Iterable[float], groups: Iterable) -> float:
    """n / design_effect(values, groups): the number of independent observations the clustered
    sample is worth (0.0 for no values)."""
    xs = list(values)
    return len(xs) / design_effect(xs, groups) if xs else 0.0


def clustered_t_stat(values: Iterable[float], groups: Iterable) -> float:
    """t_stat_mean with the cluster-robust standard error: t_stat_mean / sqrt(design_effect)."""
    xs = list(values)
    return t_stat_mean(xs) / math.sqrt(design_effect(xs, groups))


def bootstrap_mean_ci(values: Iterable[float], level: float = 0.95, n_boot: int = 2000,
                      seed: int = 7, groups: Iterable | None = None) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (resampling with replacement, random.Random(seed)).

    With `groups` (one cluster label per value) whole clusters are resampled (cluster bootstrap):
    each replicate draws G of the G clusters with replacement and takes sum / count of the drawn
    values, so values that move together are never treated as independent. Clusters are taken
    in order of first appearance. Deterministic for a given (values, groups, level, n_boot, seed).
    Empty input -> (0.0, 0.0).
    """
    xs = list(values)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    if groups is None:
        means = sorted(math.fsum(rng.choices(xs, k=n)) / n for _ in range(max(1, n_boot)))
    else:
        clusters: dict = {}
        for x, g in zip(xs, _labels(xs, groups)):
            clusters.setdefault(g, []).append(x)
        units = [(math.fsum(c), len(c)) for c in clusters.values()]
        k = len(units)
        means = []
        for _ in range(max(1, n_boot)):
            pick = rng.choices(units, k=k)
            means.append(math.fsum(u[0] for u in pick) / sum(u[1] for u in pick))
        means.sort()
    alpha = (1.0 - level) / 2.0
    return _quantile(means, alpha), _quantile(means, 1.0 - alpha)


def confidence_sequence(values: Iterable[float], level: float = 0.95, groups: Iterable | None = None,
                        target_n: int = 200) -> tuple[float, float]:
    """Anytime-valid two-sided confidence sequence (CS) for the mean: the Gaussian-mixture
    asymptotic CS (Robbins 1970; Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas, "Time-uniform
    central limit theory and asymptotic confidence sequences", Annals of Statistics 2024, Thm 2.2):

        mean +/- sd * sqrt( 2 (t r + 1) / (t^2 r) * log( sqrt(t r + 1) / alpha ) )

    with alpha = 1 - level, sd the sample stdev, t the number of independent observations
    (len(values), or effective_n(values, groups) when groups are given, so clustered values widen
    the CS like a design effect and copies add nothing) and r = (-2 log alpha + log(1 - 2 log
    alpha)) / target_n (the CS is narrowest near t = target_n).

    Unlike a fixed-sample CI it may be recomputed after every new value: the probability that the
    true mean EVER leaves the sequence is about alpha, so a rule "lower end > 0" can be checked
    continuously without inflating its false-positive rate (a fixed 95% CI re-checked after every
    trade eventually excludes 0 for a zero-edge strategy). Each end of a level-L CS is a one-sided
    anytime bound with error at most 1 - L (about half that in practice). Empty input ->
    (0.0, 0.0); one value, or level >= 1 -> (-CAP, CAP). ValueError if groups has another length.
    """
    xs = list(values)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    t = float(n) if groups is None else max(1.0, effective_n(xs, _labels(xs, groups)))
    alpha = 1.0 - level
    if n < 2 or alpha <= 0:
        return -CAP, CAP
    mu = math.fsum(xs) / n
    sd = math.sqrt(math.fsum((v - mu) ** 2 for v in xs) / (n - 1))
    la = -2.0 * math.log(min(alpha, 1.0))
    r = max(la + math.log1p(la), 1e-12) / max(1, target_n)
    x = t * r
    half = sd * math.sqrt(2.0 * (x + 1.0) / (t * x) * max(0.0, math.log(math.sqrt(x + 1.0) / alpha)))
    return max(-CAP, mu - half), min(CAP, mu + half)


def skew_kurt(values: Iterable[float]) -> tuple[float, float]:
    """Population skewness m3/m2^1.5 and NON-excess kurtosis m4/m2^2 (normal = 3).

    Fewer than 2 values or a constant series -> (0.0, 3.0).
    """
    xs = list(values)
    if len(xs) < 2 or max(xs) == min(xs):
        return 0.0, 3.0
    mu = fmean(xs)
    m2 = fmean((x - mu) ** 2 for x in xs)
    m3 = fmean((x - mu) ** 3 for x in xs)
    m4 = fmean((x - mu) ** 4 for x in xs)
    return m3 / m2 ** 1.5, m4 / m2 ** 2


def t_stat_mean(values: Iterable[float]) -> float:
    """One-sample t statistic of the mean against 0: mean / (s / sqrt(n)); 0.0 if n < 2."""
    xs = list(values)
    if len(xs) < 2:
        return 0.0
    return _ratio(fmean(xs), stdev(xs) / math.sqrt(len(xs)))


def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012, "The Sharpe Ratio Efficient
    Frontier", Journal of Risk 15(2)):

        PSR(SR*) = Phi( (SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    with SR the non-annualized Sharpe of n observations, g3 the skewness and g4 the NON-excess
    kurtosis. n may be an effective (float) sample size, e.g. effective_n() of clustered trades.
    A non-positive radicand is replaced by 1e-12; n <= 1 gives Phi(0) = 0.5.
    """
    radicand = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    z = (sr - sr_star) * math.sqrt(max(n - 1, 0)) / math.sqrt(max(radicand, 1e-12))
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe of n_trials unskilled trials (Bailey & Lopez de Prado, 2014,
    "The Deflated Sharpe Ratio", Journal of Portfolio Management 40(5), eq. for SR_0):

        E[max SR] = sqrt(V[SR]) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))

    with g the Euler-Mascheroni constant. 0.0 if N <= 1 or var_sr <= 0.
    """
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    inv = _NORMAL.inv_cdf
    return math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * inv(1.0 - 1.0 / n_trials)
                                + EULER_GAMMA * inv(1.0 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(values: Iterable[float], n_trials: int, var_sr: float | None = None,
                    n_eff: float | None = None) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the PSR of the per-trade Sharpe of
    `values` (mean / sample stdev, population skew and kurtosis) against
    SR* = expected_max_sharpe(n_trials, var_sr), computed with n_eff observations (default
    len(values); clipped to [1, len(values)]) so that clustered trades (see effective_n) do not
    overstate the evidence. var_sr defaults to 1/n_eff (the null variance of a Sharpe estimate),
    also when given as <= 0. 0.0 if fewer than 3 values.
    """
    xs = list(values)
    if len(xs) < 3:
        return 0.0
    n = float(len(xs)) if n_eff is None else max(1.0, min(float(n_eff), float(len(xs))))
    if var_sr is None or var_sr <= 0:
        var_sr = 1.0 / n
    skew, kurt = skew_kurt(xs)
    return probabilistic_sharpe(_sharpe(xs), n, skew, kurt, expected_max_sharpe(n_trials, var_sr))
