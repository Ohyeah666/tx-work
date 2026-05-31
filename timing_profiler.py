import time
from contextlib import contextmanager


class TimingProfiler:
    def __init__(self, enabled=False, print_every=0, prefix="timing"):
        self.enabled = bool(enabled)
        self.print_every = max(int(print_every), 0)
        self.prefix = prefix
        self.totals = {}
        self.counts = {}
        self.last = {}
        self.events = 0

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start_time)

    def record(self, name, elapsed_seconds):
        self.totals[name] = self.totals.get(name, 0.0) + elapsed_seconds
        self.counts[name] = self.counts.get(name, 0) + 1
        self.last[name] = elapsed_seconds

    def snapshot(self):
        return {
            name: {
                "total": self.totals[name],
                "count": self.counts[name],
                "avg": self.totals[name] / self.counts[name],
                "last": self.last[name],
            }
            for name in sorted(self.totals)
        }

    def format_summary(self):
        parts = []
        for name, stats in self.snapshot().items():
            parts.append(
                f"{name}: last={stats['last']:.4f}s avg={stats['avg']:.4f}s count={stats['count']}"
            )
        return " | ".join(parts)

    def maybe_print(self, label=None):
        if not self.enabled or self.print_every <= 0:
            return

        self.events += 1
        if self.events % self.print_every != 0:
            return

        label_text = f" {label}" if label else ""
        print(f"[{self.prefix}{label_text}] {self.format_summary()}")
