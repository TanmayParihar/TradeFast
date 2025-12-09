"""
Training progress display with rich library.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_acc: float = 0.0
    val_acc: float = 0.0
    learning_rate: float = 0.0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    patience: int = 10
    gpu_memory_mb: float = 0.0
    gpu_memory_pct: float = 0.0
    batch_time_ms: float = 0.0
    epoch_time_s: float = 0.0
    extra: dict = field(default_factory=dict)


class TrainingProgress:
    """Rich-based training progress display."""

    def __init__(
        self,
        total_epochs: int,
        model_name: str = "Model",
        show_gpu: bool = True,
        refresh_rate: int = 10,
    ):
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.show_gpu = show_gpu and torch.cuda.is_available()
        self.refresh_rate = refresh_rate

        self.console = Console()
        self.metrics = TrainingMetrics(total_epochs=total_epochs)
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        self._epoch_start_time = 0.0
        self._batch_times: list[float] = []

    def _create_metrics_table(self) -> Table:
        """Create metrics display table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Row 1: Losses
        table.add_row(
            "Train Loss",
            f"{self.metrics.train_loss:.6f}",
            "Val Loss",
            f"{self.metrics.val_loss:.6f}",
        )

        # Row 2: Accuracies
        table.add_row(
            "Train Acc",
            f"{self.metrics.train_acc:.2%}",
            "Val Acc",
            f"{self.metrics.val_acc:.2%}",
        )

        # Row 3: Learning rate and best loss
        best_indicator = "✓" if self.metrics.val_loss <= self.metrics.best_val_loss else ""
        table.add_row(
            "Learning Rate",
            f"{self.metrics.learning_rate:.2e}",
            "Best Val Loss",
            f"{self.metrics.best_val_loss:.6f} {best_indicator}",
        )

        # Row 4: Early stopping and timing
        patience_str = f"{self.metrics.patience_counter}/{self.metrics.patience}"
        table.add_row(
            "Early Stop",
            patience_str,
            "Epoch Time",
            f"{self.metrics.epoch_time_s:.1f}s",
        )

        # Row 5: GPU info
        if self.show_gpu:
            table.add_row(
                "GPU Memory",
                f"{self.metrics.gpu_memory_mb:.0f} MB",
                "GPU Util",
                f"{self.metrics.gpu_memory_pct:.1f}%",
            )

        # Extra metrics
        for key, value in self.metrics.extra.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}", "", "")

        return table

    def _create_progress_display(self, batch_progress: Progress | None = None) -> Panel:
        """Create the full progress display."""
        layout = Layout()

        # Header
        epoch_str = f"Epoch {self.metrics.epoch}/{self.total_epochs}"
        header = Text(f" {self.model_name} - {epoch_str} ", style="bold white on blue")

        # Metrics table
        metrics_table = self._create_metrics_table()

        # Mini history chart (last 10 epochs)
        history_str = self._create_mini_chart()

        # Combine into panel
        content = Table.grid(padding=1)
        content.add_row(metrics_table)
        if history_str:
            content.add_row(Text(history_str, style="dim"))
        if batch_progress:
            content.add_row(batch_progress)

        return Panel(content, title=str(header), border_style="blue")

    def _create_mini_chart(self) -> str:
        """Create a mini ASCII chart of recent losses."""
        if len(self.history["train_loss"]) < 2:
            return ""

        train = self.history["train_loss"][-10:]
        val = self.history["val_loss"][-10:]

        if not train or not val:
            return ""

        # Normalize to 0-1 range
        all_vals = train + val
        min_val = min(all_vals)
        max_val = max(all_vals)
        range_val = max_val - min_val if max_val > min_val else 1

        def normalize(v):
            return (v - min_val) / range_val

        # Create sparkline
        blocks = " ▁▂▃▄▅▆▇█"
        train_line = "".join(blocks[int(normalize(v) * 8)] for v in train)
        val_line = "".join(blocks[int(normalize(v) * 8)] for v in val)

        return f"Loss trend: Train [{train_line}] Val [{val_line}]"

    def start_epoch(self, epoch: int) -> None:
        """Called at start of epoch."""
        self.metrics.epoch = epoch
        self._epoch_start_time = time.time()
        self._batch_times = []

    def update_batch(self, batch_loss: float) -> None:
        """Update after each batch."""
        self._batch_times.append(time.time())

    def end_epoch(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float = 0.0,
        val_acc: float = 0.0,
        learning_rate: float = 0.0,
        patience_counter: int = 0,
        **extra: Any,
    ) -> None:
        """Update metrics at end of epoch."""
        self.metrics.train_loss = train_loss
        self.metrics.val_loss = val_loss
        self.metrics.train_acc = train_acc
        self.metrics.val_acc = val_acc
        self.metrics.learning_rate = learning_rate
        self.metrics.patience_counter = patience_counter
        self.metrics.epoch_time_s = time.time() - self._epoch_start_time
        self.metrics.extra = extra

        if val_loss < self.metrics.best_val_loss:
            self.metrics.best_val_loss = val_loss

        # Update GPU metrics
        if self.show_gpu:
            self._update_gpu_metrics()

        # Update history
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(learning_rate)

    def _update_gpu_metrics(self) -> None:
        """Update GPU memory metrics."""
        if torch.cuda.is_available():
            self.metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            self.metrics.gpu_memory_pct = (self.metrics.gpu_memory_mb / total_memory) * 100

    def display(self) -> None:
        """Display current progress (one-shot)."""
        self.console.print(self._create_progress_display())

    def print_summary(self) -> None:
        """Print training summary."""
        self.console.print()
        self.console.rule("[bold blue]Training Complete")

        summary = Table(title="Training Summary", show_header=True)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Final", style="green")
        summary.add_column("Best", style="yellow")

        summary.add_row(
            "Val Loss",
            f"{self.metrics.val_loss:.6f}",
            f"{self.metrics.best_val_loss:.6f}",
        )
        summary.add_row(
            "Val Acc",
            f"{self.metrics.val_acc:.2%}",
            f"{max(self.history['val_acc']):.2%}" if self.history['val_acc'] else "N/A",
        )
        summary.add_row(
            "Train Loss",
            f"{self.metrics.train_loss:.6f}",
            f"{min(self.history['train_loss']):.6f}" if self.history['train_loss'] else "N/A",
        )
        summary.add_row(
            "Epochs",
            f"{self.metrics.epoch}",
            f"{self.total_epochs}",
        )

        self.console.print(summary)
        self.console.print()


class BatchProgress:
    """Progress bar for batches within an epoch."""

    def __init__(self, total_batches: int, description: str = "Training"):
        self.total_batches = total_batches
        self.description = description
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[loss]:.4f}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self.task_id = None

    def __enter__(self):
        self.progress.start()
        self.task_id = self.progress.add_task(
            self.description,
            total=self.total_batches,
            loss=0.0,
        )
        return self

    def __exit__(self, *args):
        self.progress.stop()

    def update(self, loss: float = 0.0) -> None:
        """Update progress bar."""
        self.progress.update(self.task_id, advance=1, loss=loss)


def create_training_callback(
    total_epochs: int,
    model_name: str = "Model",
    show_gpu: bool = True,
):
    """
    Create a training callback function for use with model training.

    Returns a tuple of (progress, callback_fn).

    Usage:
        progress, callback = create_training_callback(100, "MyModel")
        for epoch in range(100):
            progress.start_epoch(epoch + 1)
            # ... training loop ...
            callback(train_loss, val_loss, train_acc, val_acc, lr)
            progress.display()
        progress.print_summary()
    """
    progress = TrainingProgress(total_epochs, model_name, show_gpu)

    def callback(
        train_loss: float,
        val_loss: float,
        train_acc: float = 0.0,
        val_acc: float = 0.0,
        learning_rate: float = 0.0,
        patience_counter: int = 0,
        **extra: Any,
    ):
        progress.end_epoch(
            train_loss, val_loss, train_acc, val_acc,
            learning_rate, patience_counter, **extra
        )

    return progress, callback


class LiveTrainingDisplay:
    """Context manager for live updating training display."""

    def __init__(
        self,
        total_epochs: int,
        model_name: str = "Model",
        show_gpu: bool = True,
        refresh_rate: int = 4,
    ):
        self.progress = TrainingProgress(total_epochs, model_name, show_gpu)
        self.refresh_rate = refresh_rate
        self.live = None

    def __enter__(self):
        self.live = Live(
            self.progress._create_progress_display(),
            refresh_per_second=self.refresh_rate,
            console=self.progress.console,
        )
        self.live.start()
        return self.progress

    def __exit__(self, *args):
        if self.live:
            self.live.stop()
        self.progress.print_summary()

    def update(self) -> None:
        """Update the live display."""
        if self.live:
            self.live.update(self.progress._create_progress_display())
