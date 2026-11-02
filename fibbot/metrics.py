"""Pure performance statistics for trade lists and equity curves.

Conventions: a trade is a *win* when ret_pct > 0 and a *loss* otherwise (ret_pct <= 0). Ratios
whose denominator is zero (profit factor without losses, Sharpe of a constant series, ...) and
ratios beyond +/-CAP are reported as +/-CAP (or 0.0 when the numerator is 0), which keeps every
value finite and JSON-safe. Standard deviations are sample (n-1) estimates; skew and kurtosis are
population moments, as in the Bailey & Lopez de Prado formulas.
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

    Trades are ordered by entry_t (stable) for max_consec_losses; profit_factor is computed on
    ret_pct; sharpe_per_trade = mean / sample stdev of r_multiple. Empty input -> all zeros.
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
    gross_win = math.fsum(tr.ret_pct for tr in ts if tr.ret_pct > 0)
    gross_loss = -math.fsum(tr.ret_pct for tr in ts if tr.ret_pct < 0)
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


def bootstrap_mean_ci(values: Iterable[float], level: float = 0.95, n_boot: int = 2000,
                      seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (resampling with replacement, random.Random(seed)).

    Deterministic for a given (values, level, n_boot, seed). Empty input -> (0.0, 0.0).
    """
    xs = list(values)
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(xs, k=n)) / n for _ in range(max(1, n_boot)))
    alpha = (1.0 - level) / 2.0
    return _quantile(means, alpha), _quantile(means, 1.0 - alpha)


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
    kurtosis. A non-positive radicand is replaced by 1e-12; n <= 1 gives Phi(0) = 0.5.
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


def deflated_sharpe(values: Iterable[float], n_trials: int, var_sr: float | None = None) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the PSR of the per-trade Sharpe of
    `values` (mean / sample stdev, population skew and kurtosis) against
    SR* = expected_max_sharpe(n_trials, var_sr). var_sr defaults to 1/len(values) (the null
    variance of a Sharpe estimate), also when given as <= 0. 0.0 if fewer than 3 values.
    """
    xs = list(values)
    if len(xs) < 3:
        return 0.0
    if var_sr is None or var_sr <= 0:
        var_sr = 1.0 / len(xs)
    skew, kurt = skew_kurt(xs)
    return probabilistic_sharpe(_sharpe(xs), len(xs), skew, kurt, expected_max_sharpe(n_trials, var_sr))
