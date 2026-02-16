import logging
import time
from typing import Dict, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self, total_iters: int, log_interval: int = 100):
        self.total_iters = total_iters
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.current_iter = 0

    def update(self, batch_idx: int, loss: float, lr: float) -> Optional[Dict]:
        self.current_iter += 1
        current_time = time.time()

        if batch_idx % self.log_interval != 0 and current_time - self.last_log_time < 10:
            return None

        elapsed = current_time - self.start_time
        iters_per_sec = self.current_iter / max(elapsed, 1e-8)
        remaining_iters = self.total_iters - self.current_iter
        eta_seconds = remaining_iters / max(iters_per_sec, 1e-8)
        self.last_log_time = current_time

        # Format ETA for display
        if eta_seconds > 3600:
            eta_str = f"{eta_seconds/3600:.1f}h"
        elif eta_seconds > 60:
            eta_str = f"{eta_seconds/60:.1f}m"
        else:
            eta_str = f"{eta_seconds:.0f}s"

        return {
            'iter': self.current_iter,
            'loss': loss,
            'lr': lr,
            'speed': iters_per_sec,
            'eta': eta_str,
            'eta_seconds': eta_seconds,
            'progress': (self.current_iter / self.total_iters) * 100
        }

class RichProgressDisplay:
    """Renders training progress using rich"""

    def __init__(self):
        # Use stderr for progress bar so it doesn't interfere with stdout logs
        self.console = Console(stderr=True)
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("• Loss: {task.fields[loss]:.4f}"),
            TextColumn("• LR: {task.fields[lr]:.2e}"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=10  # Faster refresh
        )
        self.task = None
        self.started = False

    def create_task(self, total_steps: int, description: str = "[cyan]Training"):
        """Create a new progress task"""
        self.task = self.progress.add_task(
            description,
            total=total_steps,
            loss=0,
            lr=0
        )

    def start(self):
        """Start the progress display"""
        if not self.started:
            self.progress.start()
            self.started = True
            # Don't print an extra blank line here

    def update(self, progress_data: Dict):
        """Update display with data from ProgressTracker"""
        if self.task is not None:
            self.progress.update(
                self.task,
                advance=1,
                loss=progress_data.get('loss', 0),
                lr=progress_data.get('lr', 0)
            )

    def stop(self):
        """Stop the progress display"""
        if self.started:
            self.progress.stop()
            self.started = False
            # Add a newline to separate from next output
            self.console.print()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
