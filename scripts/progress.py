import sys
import threading
import time


class ProgressTracker:
    def __init__(self, total: int, label: str = "Progress", initial: int = 0):
        self.total = max(0, total)
        self.label = label
        self.completed = max(0, min(initial, self.total))
        self.started_at = time.time()
        self.last_render_at = 0.0
        self.lock = threading.Lock()

    def advance(self, step: int = 1, current: str = "", force: bool = False) -> None:
        with self.lock:
            self.completed = min(self.total, self.completed + step)
            self._render(current=current, force=force)

    def render(self, current: str = "", force: bool = False) -> None:
        with self.lock:
            self._render(current=current, force=force)

    def finish(self, current: str = "done") -> None:
        with self.lock:
            self.completed = self.total
            self._render(current=current, force=True)
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _render(self, current: str = "", force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_render_at < 0.5 and self.completed < self.total:
            return
        self.last_render_at = now

        elapsed = max(0.001, now - self.started_at)
        rate = self.completed / elapsed if self.completed else 0.0
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else None
        estimated_total = elapsed + eta if eta is not None else None
        bar = self._bar()

        message = (
            f"\r{self.label} "
            f"{self.completed}/{self.total} "
            f"{bar} "
            f"{self._format_seconds(elapsed)}/{self._format_seconds(estimated_total)}"
        )

        sys.stdout.write(message)
        sys.stdout.flush()

    def _bar(self, width: int = 24) -> str:
        percent = (self.completed / self.total) if self.total else 1.0
        filled = int(width * percent)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    @staticmethod
    def _format_seconds(value: float | None) -> str:
        if value is None:
            return "--:--"
        total_seconds = int(max(0, value))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
