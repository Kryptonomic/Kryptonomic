"""Human-readable reports (see DESIGN.md, "fibbot/report.py").

render_status() is compact terminal text; write_report() writes home/REPORT.md, Markdown that reads
well on GitHub and on a phone. Both show the same lab state: roster and paper accounts, graduation
checks, recent paper trades, the value of the brain's signal reviews, the latest research and
review journal entries and the head of the brain notebook.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import home_path
from .data.candles import atomic_write
from .lab import NOTEBOOK_HEADER, Lab

DISCLAIMER = "Paper trading only: fibbot never places real orders. Nothing here is financial advice."
RECENT_TRADES = 15
JOURNAL_ENTRIES = 3
JOURNAL_BODY_CHARS = 800
NOTEBOOK_LINES = 60
REVIEW_GROUPS = ("take", "skip", "unreviewed")


def _when(t: int | None) -> str:
    """'YYYY-MM-DD HH:MM' UTC, or '-'."""
    return "-" if t is None else datetime.fromtimestamp(int(t), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _f(x, fmt: str = "{:.4g}") -> str:
    """Format a number, '-' for None."""
    return "-" if x is None else fmt.format(x)


def _cell(x) -> str:
    """Markdown table cell text: pipes escaped, newlines flattened."""
    return str(x).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list], right: set[int] = frozenset()) -> list[str]:
    """Markdown table lines; columns whose index is in `right` are right-aligned."""
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("--:" if i in right else "---" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out


def _clip(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + " [...]"


def _notebook_head(lab: Lab) -> list[str]:
    """First NOTEBOOK_LINES lines of the notebook after its "# Brain notebook" header."""
    lines = lab.notebook().splitlines()
    if lines and lines[0].strip() == NOTEBOOK_HEADER:
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return lines[:NOTEBOOK_LINES]


def _research_journal(lab: Lab) -> list[dict]:
    """Newest JOURNAL_ENTRIES journal entries of kind research or review."""
    rows = lab.store.journal(JOURNAL_ENTRIES, "research") + lab.store.journal(JOURNAL_ENTRIES, "review")
    return sorted(rows, key=lambda r: (r["t"], r["id"]), reverse=True)[:JOURNAL_ENTRIES]


def _names(lab: Lab) -> dict[str, str]:
    return {r["strategy_id"]: r["name"] for r in lab.store.roster(None)}


def _account_cells(e: dict) -> list:
    a = e["account"]
    return [_f(a["days"], "{:.1f}"), f"{a['n_closed']} / {a['n_open']}", _f(a["avg_r"], "{:+.2f}"),
            _f(a["profit_factor"], "{:.2f}"), _f(a["return"], "{:+.1%}"), _f(a["max_drawdown"], "{:.1%}")]


def _trade_cells(r: dict, names: dict[str, str]) -> list:
    is_open = r["exit_t"] is None
    pnl = ("-" if r["notional"] is None else "not taken" if r["notional"] == 0   # 0: risk limits
           else _f(r["pnl"], "{:+.2f}"))
    return [_when(r["entry_t"]), names.get(r["strategy_id"], r["strategy_id"]), r["symbol"],
            "long" if r["side"] > 0 else "short", _f(r["entry_price"], "{:.6g}"),
            _f(r["exit_price"], "{:.6g}") + (" (mark)" if is_open else ""),
            "open" if is_open else r["exit_reason"], _f(r["r_multiple"], "{:+.2f}"), pnl]


def _check_line(c: dict) -> str:
    """GitHub task-list item: [x] passed, [ ] not yet."""
    detail = c.get("note") or f"{_f(c['value'])} (threshold {_f(c['threshold'])})"
    return f"- [{'x' if c['passed'] else ' '}] {c['name']}: {detail}"


def _data_lines(st: dict) -> list[str]:
    """Universe, candle ranges and research windows per timeframe."""
    lines = [f"universe ({len(st['universe'])}): {', '.join(st['universe']) or '-'}"]
    for tf, d in st["data"].items():
        w = st["windows"].get(tf, {})
        win = (w["error"] if "error" in w else
               f"research {w['research'][0]}..{w['research'][1]}, validation {w['validation'][0]}..{w['validation'][1]}")
        lines.append(f"{tf} candles: {d['symbols']} symbols {d['start']}..{d['end']}; {win}")
    if not st["data"]:
        lines.append("no candles cached yet (python -m fibbot data)")
    return lines


def _budget_line(st: dict) -> str:
    b = st["budget"]
    if st["cycle"] is None:
        return "budget: unlimited (no brain cycle running)"
    return (f"budget ({st['cycle']} cycle): {b['backtest_units_left']} backtest units,"
            f" {b['validations_left']} validations left")


def _activity_line(st: dict) -> str:
    ts, ex, last = st["trial_stats"], st["experiments"], st["last"]
    kinds = ", ".join(f"{k} {v}" for k, v in ex.items() if k != "total" and v)
    return (f"trials {ts['n']} (var Sharpe {_f(ts['var_sharpe'])}); experiments {ex['total']}"
            + (f" ({kinds})" if kinds else "")
            + f"; last research {last['research'] or '-'}, review {last['review'] or '-'},"
              f" sweep {last['sweep'] or '-'}")


def render_status(lab: Lab, now_t: int | None = None) -> str:
    """Compact plain-text status for the terminal."""
    st = lab.status(now_t=now_t)
    slots = st["roster_slots"]
    lines = [f"fibbot status {st['as_of']} | signal review: {st['signal_review']}", DISCLAIMER,
             *_data_lines(st), f"roster {slots['paper']}/{slots['max']} paper slots:"]
    for e in st["roster"]:
        a, g = e["account"], e["graduation"]
        lines.append(f"  {e['name']} [{e['strategy_id']}] {e['timeframe']} {e['status']}:"
                     f" {a['days']:.1f}d, closed {a['n_closed']} open {a['n_open']},"
                     f" mean R {a['avg_r']:+.2f}, PF {a['profit_factor']:.2f}, return {a['return']:+.1%},"
                     f" DD {a['max_drawdown']:.1%} -> {g['status']}"
                     + (f" (not yet: {', '.join(g['failed'])})" if g["failed"] else ""))
    if not st["roster"]:
        lines.append("  (empty)")
    lines += [f"  retired {r['name']} [{r['strategy_id']}] {r['since']}: {r['reason']}" for r in st["retired"]]
    names = _names(lab)
    trades = lab.store.paper_trades(limit=5)
    if trades:
        lines.append("recent paper trades:")
        lines += ["  " + " ".join(str(c) for c in _trade_cells(r, names)) for r in trades]
    rv = lab.signal_review_stats()
    lines.append(f"signal review ({rv['mode']}): "
                 + " | ".join(f"{g} n {rv[g]['n']}" + (f" mean R {rv[g]['avg_r']:+.2f}" if rv[g]["n"] else "")
                              for g in REVIEW_GROUPS)
                 + f" | skip-take {_f(rv['skip_minus_take_avg_r'], '{:+.2f}')}")
    lines += [_activity_line(st), _budget_line(st)]
    lines += [f"{_when(j['t'])} {j['kind']}: {j['title']}" for j in _research_journal(lab)[:1]]
    return "\n".join(lines)


def _markdown(lab: Lab, st: dict, now_t: int | None) -> str:
    slots, names = st["roster_slots"], _names(lab)
    md = ["# fibbot report", "",
          f"**As of** {st['as_of']} · {len(st['universe'])} coins · signal review: {st['signal_review']}",
          "", f"> **{DISCLAIMER}**", "", f"## Paper roster ({slots['paper']}/{slots['max']} slots)", ""]
    if st["roster"]:
        md += _table(["Strategy", "TF", "Days", "Closed / open", "Mean R", "PF", "Return", "Max DD", "Graduation"],
                     [[f"{e['name']} `{e['strategy_id']}`", e["timeframe"], *_account_cells(e),
                       e["graduation"]["status"]] for e in st["roster"]], right={2, 3, 4, 5, 6, 7})
    else:
        md.append("_No strategies on paper yet._")
    if st["retired"]:
        md += ["", "Retired: " + "; ".join(f"{r['name']} `{r['strategy_id']}` ({r['since']}): {r['reason']}"
                                           for r in st["retired"])]

    md += ["", "## Graduation checks", "",
           "Graduation is a statistical verdict on paper results. It never enables live trading."]
    for e in st["roster"]:
        g = lab.graduation(e["strategy_id"], now_t=now_t)
        md += ["", f"### {e['name']} · {g['status']}", "", g["summary"], "",
               *(_check_line(c) for c in g["checks"])]

    trades = lab.store.paper_trades(limit=RECENT_TRADES)
    md += ["", f"## Recent paper trades (last {RECENT_TRADES})", ""]
    md += (_table(["Entry (UTC)", "Strategy", "Symbol", "Side", "Entry", "Exit", "Reason", "R", "PnL"],
                  [_trade_cells(r, names) for r in trades], right={4, 5, 7, 8})
           if trades else ["_No paper trades yet._"])

    rv = lab.signal_review_stats()
    md += ["", f"## Brain signal review (mode: {rv['mode']})", "",
           f"{rv['n_signals_reviewed']} of {rv['n_signals']} signals reviewed. Closed paper trades by"
           " the brain's decision on their setup:", ""]
    md += _table(["Brain decision", "Closed trades", "Mean R", "Total R", "Win rate", "PF"],
                 [[g, rv[g]["n"], *(["-"] * 4 if not rv[g]["n"] else    # no trades: no statistics
                                    [_f(rv[g]["avg_r"], "{:+.2f}"), _f(rv[g]["total_r"], "{:+.2f}"),
                                     _f(rv[g]["win_rate"], "{:.0%}"), _f(rv[g]["profit_factor"], "{:.2f}")])]
                  for g in REVIEW_GROUPS], right={1, 2, 3, 4, 5})
    md += ["", f"Skip minus take mean R: {_f(rv['skip_minus_take_avg_r'], '{:+.3f}')} (above 0 means the"
               " setups the brain would skip did better than the ones it would take)."]

    md += ["", "## Latest research and reviews"]
    entries = _research_journal(lab)
    for j in entries:
        md += ["", f"### {_when(j['t'])} UTC · {j['kind']} · {j['title']}", "",
               *(f"> {line}" for line in _clip(j["body"], JOURNAL_BODY_CHARS).splitlines())]
    if not entries:
        md += ["", "_No research or review entries yet._"]

    md += ["", f"## Brain notebook (first {NOTEBOOK_LINES} lines)", ""]
    md += [f"> {ln}" for ln in _notebook_head(lab)] or ["_Empty._"]

    md += ["", "## Data and research activity", "",
           *(f"- {ln}" for ln in (*_data_lines(st), _activity_line(st), _budget_line(st))), ""]
    return "\n".join(md)


def write_report(lab: Lab, path=None, now_t: int | None = None) -> Path:
    """Write the Markdown report to `path` (default home/REPORT.md) atomically; returns the path."""
    p = Path(path) if path is not None else home_path(lab.cfg, "REPORT.md")
    atomic_write(p, _markdown(lab, lab.status(now_t=now_t), now_t).encode("utf-8"))
    return p
