"""Visualization utilities for multimodal search results."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()


def render_results_table(
    results: Sequence[dict[str, Any]],
    title: str = "Search Results",
) -> None:
    """Print a rich terminal table with search results and similarity scores."""
    if not results:
        console.print("[yellow]No matching items found.[/yellow]")
        return

    table = Table(title=f"🔍 {title}", show_header=True, header_style="bold cyan")
    table.add_column("Rank", justify="center", style="dim", width=6)
    table.add_column("Similarity", justify="center", style="bold green")
    table.add_column("Kind", justify="center", style="magenta")
    table.add_column("Filename", style="bold white")
    table.add_column("Path", style="dim", overflow="ellipsis")

    for rank, item in enumerate(results, start=1):
        sim = item.get("similarity", 0.0)
        # Visual color bar
        bar_len = int(sim * 10)
        bar = "█" * max(0, min(10, bar_len))
        table.add_row(
            str(rank),
            f"{sim:.3f} [green]{bar}[/green]",
            item.get("kind", "-"),
            item.get("filename", "-"),
            item.get("path", "-"),
        )

    console.print(table)


def plot_image_results(
    results: Sequence[dict[str, Any]],
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Display retrieved image results side-by-side using matplotlib.

    Mirrors the show_images() function from the DataCamp tutorial.
    """
    image_items = [
        r for r in results if r.get("kind") == "image" and Path(r.get("path", "")).exists()
    ]
    if not image_items:
        console.print("[yellow]No valid images in results to display.[/yellow]")
        return

    n = len(image_items)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for idx, (ax, item) in enumerate(zip(axes[0], image_items)):
        img_path = Path(item["path"])
        score = item.get("similarity", 0.0)
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f"Rank {idx + 1}: {img_path.name}\nSimilarity: {score:.3f}", fontsize=11)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        console.print(f"[green]Saved plot to:[/green] {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
