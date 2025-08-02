import time
import pytest


class TestsuiteResult:
    def __init__(self):
        self.reports = []
        self.exitcode = 0
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.xfailed = 0
        self.skipped = 0
        self.duration = 0

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if report.when == "call":
            # object { nodeid: testcase, outcome: passed, etc.}
            self.reports.append(report)

    def pytest_collection_modifyitems(self, items):
        self.total = len(items)

    def pytest_terminal_summary(self, terminalreporter, exitstatus):
        self.exitcode = int(exitstatus)
        self.passed = len(terminalreporter.stats.get("passed", []))
        self.failed = len(terminalreporter.stats.get("failed", []))
        self.xfailed = len(terminalreporter.stats.get("xfailed", []))
        self.skipped = len(terminalreporter.stats.get("skipped", []))
        self.duration = time.time() - terminalreporter._sessionstarttime

