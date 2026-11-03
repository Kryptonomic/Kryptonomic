"""Tests for fibbot.report: the Markdown report (sections, tables, check marks, truncation,
escaping, default and custom paths) and the compact terminal status."""
import re
import unittest

from fibbot.lab import NOTEBOOK_HEADER, iso_time
from fibbot.report import DISCLAIMER, JOURNAL_BODY_CHARS, NOTEBOOK_LINES, RECENT_TRADES, render_status, write_report
from tests.test_lab import H4, SPEC, SYMS, FakeCandles, LabCase, mk_signal, mk_trade, record_validation

NOW = 1_700_000_123
NAME = "deep|pullbacks"


def section(text: str, heading: str) -> str:
    """Body of the '## heading...' section up to the next '## ' heading."""
    return re.search(rf"^## {re.escape(heading)}.*?\n(.*?)(?=^## |\Z)", text, re.M | re.S).group(1)


class ReportCase(LabCase):
    def setUp(self):
        super().setUp()
        # validated just before the promotion at NOW (years ago in real time, so it does not
        # count against research.max_validations_per_week in the budget lines below)
        record_validation(self.store, SPEC, avg_r=0.3, now_t=NOW - 60)
        self.sid = self.lab.promote(SPEC, NAME, "test", now_t=NOW)["strategy_id"]
        start = self.store.get_strategy(self.sid)["paper_start_t"]
        for k in range(20):
            r = (1.5, -1.0)[k % 2]
            tr = mk_trade(self.sid, SYMS[k % 3], f"L{k}", start + k * H4, r)
            self.store.upsert_paper_trade(self.sid, tr, 0.0 if k == 19 else 100.0, 0.0 if k == 19 else r * 2)
            self.store.add_signal(mk_signal(self.sid, tr.symbol, tr.leg_id, tr.entry_t - H4))
            if k < 6:
                self.store.set_signal_review(self.sid, tr.symbol, tr.leg_id, ("take", "skip")[k % 2], 0.6, "x")
        self.store.upsert_paper_trade(self.sid, mk_trade(self.sid, "S0USDT", "L99", start + 30 * H4, 0.5,
                                                         closed=False), 100.0, 1.0)
        self.store.add_journal("research", "Research cycle 1", "finding " * 200, now_t=NOW + 10)
        self.store.add_journal("review", "Daily review", "all fine", now_t=NOW + 20)
        self.store.add_journal("system", "Data update failed", "boom", now_t=NOW + 30)
        self.lab.notebook_rewrite("\n".join(f"line {k}" for k in range(1, 101)))
        self.now = start + 10 * 86400


class TestWriteReport(ReportCase):
    def test_report_sections(self):
        path = write_report(self.lab, now_t=self.now)
        self.assertEqual(path, self.home / "REPORT.md")
        text = path.read_text()
        self.assertTrue(text.startswith("# fibbot report\n"))
        self.assertIn(f"**As of** {iso_time(self.now)}", text)
        self.assertIn(f"> **{DISCLAIMER}**", text)

        roster = section(text, "Paper roster")
        self.assertIn("(1/4 slots)", text)
        self.assertIn(f"| deep\\|pullbacks `{self.sid}` | 4h | 10.0 | 20 / 1 |", roster)

        checks = section(text, "Graduation checks")
        grad = self.lab.graduation(self.sid, now_t=self.now)
        self.assertIn(f"### {NAME} · {grad['status']}", checks)
        marks = re.findall(r"^- \[([x ])\] (\w+):", checks, re.M)
        self.assertEqual(marks, [("x" if c["passed"] else " ", c["name"]) for c in grad["checks"]])
        self.assertIn(("x", "max_drawdown"), marks)
        # F33: the fixture's candles end before the paper start, so no paper day counts yet
        days = next(c for c in grad["checks"] if c["name"] == "min_days")
        self.assertEqual(days["value"], 0.0)
        self.assertIn("- [ ] min_days: days up to the newest bar", checks)
        self.assertIn("10.0 calendar days", checks)

        trades = section(text, "Recent paper trades").strip().splitlines()
        self.assertEqual(len(trades), 2 + RECENT_TRADES)                    # header, rule, rows
        self.assertIn("(mark)", trades[2])                                   # the open trade is newest
        self.assertIn("| open |", trades[2])
        self.assertIn("| not taken |", trades[3])                            # skipped by the risk limits
        self.assertIn(f"| deep\\|pullbacks | S", trades[4])

        review = section(text, "Brain signal review")
        self.assertIn("(mode: shadow)", text)
        self.assertIn("6 of 20 signals reviewed", review)
        self.assertRegex(review, r"\| take \| 3 \| \+1\.50 \|")
        self.assertRegex(review, r"\| skip \| 3 \| -1\.00 \|")
        self.assertRegex(review, r"\| unreviewed \| 14 \|")
        self.assertIn("Skip minus take mean R: -2.500", review)

        journal = section(text, "Latest research and reviews")
        self.assertLess(journal.index("Daily review"), journal.index("Research cycle 1"))   # newest first
        self.assertNotIn("Data update failed", journal)
        body = re.search(r"^> (finding.*)$", journal, re.M).group(1)
        self.assertTrue(body.endswith(" [...]"))
        self.assertLessEqual(len(body), JOURNAL_BODY_CHARS + 6)

        notebook = section(text, "Brain notebook").strip().splitlines()
        self.assertEqual(notebook, [f"> line {k}" for k in range(1, NOTEBOOK_LINES + 1)])
        self.assertNotIn(f"> {NOTEBOOK_HEADER}", text)

        data = section(text, "Data and research activity")
        self.assertIn("- 4h candles: 3 symbols 2020-09-14..", data)
        self.assertIn("- budget: unlimited (no brain cycle running)", data)

    def test_custom_path_and_empty_lab(self):
        self.lab.retire(self.sid, "done")
        empty = self.make_lab(candles=FakeCandles({}))
        path = write_report(empty, path=self.home / "out" / "r.md", now_t=self.now)
        text = path.read_text()
        self.assertEqual(path, self.home / "out" / "r.md")
        self.assertIn("_No strategies on paper yet._", text)
        self.assertIn(f"Retired: {NAME} `{self.sid}`", text)
        self.assertIn("no candles cached yet", text)
        self.assertFalse(list((self.home / "out").glob(".*.tmp")))
        write_report(empty, path=path, now_t=self.now)          # overwrite in place
        self.assertEqual(path.read_text(), text)


class TestEmptyReport(LabCase):
    def test_nothing_recorded_yet(self):
        text = write_report(self.lab, now_t=NOW).read_text()
        for placeholder in ("_No strategies on paper yet._", "_No paper trades yet._",
                            "_No research or review entries yet._", "_Empty._"):
            self.assertIn(placeholder, text)
        self.assertIn("Skip minus take mean R: -", text)
        self.assertIn("| take | 0 | - | - | - | - |", text)           # no trades: no statistics
        self.assertIn("take n 0 | skip n 0 | unreviewed n 0 |", render_status(self.lab, now_t=NOW))


class TestRenderStatus(ReportCase):
    def test_compact_terminal_status(self):
        text = render_status(self.lab, now_t=self.now)
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith(f"fibbot status {iso_time(self.now)}"))
        self.assertEqual(lines[1], DISCLAIMER)
        self.assertIn("roster 1/4 paper slots:", lines)
        strategy = next(ln for ln in lines if self.sid in ln)
        self.assertIn(f"{NAME} [{self.sid}] 4h paper: 10.0d, closed 20 open 1", strategy)
        self.assertIn("not yet: min_days, min_trades", strategy)
        i = lines.index("recent paper trades:")
        self.assertEqual(len([ln for ln in lines[i + 1:i + 7] if ln.startswith("  ")]), 5)
        self.assertTrue(any(ln.startswith("signal review (shadow): take n 3") for ln in lines))
        self.assertIn("budget: unlimited (no brain cycle running)", lines)
        self.assertTrue(lines[-1].endswith("review: Daily review"))
        self.assertLess(len(text), 3000)

    def test_budget_inside_a_cycle(self):
        self.lab.begin_cycle("research")
        self.lab.backtest(SPEC)
        self.assertIn("budget (research cycle): 199 backtest units, 3 validations left",
                      render_status(self.lab, now_t=self.now).splitlines())


if __name__ == "__main__":
    unittest.main()
