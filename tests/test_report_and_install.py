from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from graphrag.entry import main as entry_main
from graphrag.skill.install import HOME_TARGETS, default_targets, install, skill_text
from graphrag.tracing import capture, save
from graphrag.tracing.report import link_source, render_markdown, report_path
from graphrag.tracing.schema import TraceEdge

ANSWER = "Sessions retry failed requests with exponential backoff up to three times."
ITEMS = [
    {"id": "doc-1", "label": "Retry policy", "content": "Sessions retry failed requests with exponential backoff up to three times."},
    {"id": "doc-2", "label": "Proxy config", "content": "Proxies are read from environment variables like HTTPS_PROXY."},
]


class ReportTests(unittest.TestCase):
    def test_used_and_unused_items_are_split_against_the_answer(self):
        trace = capture("How do sessions retry?", ITEMS, answer=ANSWER)
        text = render_markdown(trace)
        self.assertIn("**Retrieved 2, used 1.**", text)
        used, rest = text.split("## Not used", 1)
        self.assertIn("Retry policy", used)
        self.assertIn("Proxy config", rest)

    def test_without_an_answer_nothing_is_called_used(self):
        text = render_markdown(capture("q", ITEMS))
        self.assertIn("No answer was recorded", text)
        self.assertNotIn("## Used", text)

    def test_links_say_where_they_came_from(self):
        self.assertEqual(link_source("AUTHORED"), "record")
        self.assertEqual(link_source("resolves"), "record")
        self.assertEqual(link_source("MENTIONS"), "text")
        self.assertEqual(link_source("SOMETHING_ELSE"), "unknown")
        edges = [TraceEdge("doc-1", "doc-2", "MENTIONS"), TraceEdge("doc-2", "doc-1", "AUTHORED", 0.9)]
        text = render_markdown(capture("q", ITEMS, answer=ANSWER, edges=edges))
        self.assertIn("## Links (2: 1 from records, 1 from text)", text)
        self.assertIn("Retry policy —MENTIONS→ Proxy config (from text)", text)

    def test_report_command_writes_beside_the_trace(self):
        with TemporaryDirectory() as folder:
            trace_path = save(capture("q", ITEMS, answer=ANSWER), Path(folder) / "run.json")
            self.assertEqual(entry_main(["report", str(trace_path)]), 0)
            report = report_path(trace_path)
            self.assertEqual(report.name, "run.report.md")
            self.assertIn("used 1", report.read_text(encoding="utf-8"))

    def test_report_command_refuses_a_missing_file(self):
        self.assertEqual(entry_main(["report", "no-such-trace.json"]), 1)


class InstallTests(unittest.TestCase):
    def test_the_packaged_skill_names_itself(self):
        text = skill_text()
        self.assertTrue(text.startswith("---\nname: graphweave\n"))
        self.assertIn("graphweave-github-trace", text)

    def test_install_writes_one_skill_per_directory(self):
        with TemporaryDirectory() as folder:
            written = install([Path(folder) / "a", Path(folder) / "b"])
            self.assertEqual([p.relative_to(folder).as_posix() for p in written],
                             ["a/graphweave/SKILL.md", "b/graphweave/SKILL.md"])
            self.assertEqual(written[0].read_text(encoding="utf-8"), skill_text())

    def test_default_targets_follow_what_is_installed(self):
        with TemporaryDirectory() as folder:
            home = Path(folder)
            self.assertEqual(default_targets(home), ["claude"])
            (home / ".codex").mkdir()
            self.assertEqual(default_targets(home), ["codex"])
            (home / ".claude").mkdir()
            self.assertEqual(sorted(default_targets(home)), sorted(HOME_TARGETS))

    def test_install_command_with_a_directory(self):
        with TemporaryDirectory() as folder:
            self.assertEqual(entry_main(["install", "--dir", folder]), 0)
            self.assertTrue((Path(folder) / "graphweave" / "SKILL.md").is_file())


if __name__ == "__main__":
    unittest.main()
