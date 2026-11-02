"""Command Line Interface for the Gemini Embedding 2 application."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from gemini_embedding.config import settings
from gemini_embedding.indexer import index_directory
from gemini_embedding.sample_media import generate_sample_media
from gemini_embedding.search_engine import search_by_image, search_by_text
from gemini_embedding.vector_store import MultimodalVectorStore
from gemini_embedding.visualizer import plot_image_results, render_results_table

app = typer.Typer(
    name="gemini-search",
    help="Gemini Embedding 2 Multimodal Search & Study CLI",
    add_completion=False,
)
console = Console()


@app.command()
def samples(
    output_dir: Annotated[
        str,
        typer.Option(
            "--output-dir", "-o", help="Directory where sample media files will be generated"
        ),
    ] = "./media",
) -> None:
    """Generate synthetic media samples (images, audio, query image) for testing."""
    console.print(f"[bold cyan]Generating synthetic sample media in:[/bold cyan] {output_dir}")
    result = generate_sample_media(output_dir)
    console.print(f"[green]✓ Generated {len(result['images'])} sample images[/green]")
    console.print(f"[green]✓ Generated {len(result['audio'])} synthetic audio WAV files[/green]")
    console.print(f"[green]✓ Generated {len(result['queries'])} query image(s)[/green]")
    console.print(
        Panel("[bold green]Sample media ready! Next step: run 'gemini-search index'[/bold green]")
    )


@app.command()
def index(
    media_dir: Annotated[
        str,
        typer.Option("--media-dir", "-m", help="Directory containing media files to index"),
    ] = "./media",
    delay: Annotated[
        float,
        typer.Option(
            "--delay", "-d", help="Seconds to delay between API calls to stay under rate limit"
        ),
    ] = 0.5,
) -> None:
    """Index all supported media in the directory into ChromaDB with Gemini Embedding 2."""
    media_path = Path(media_dir)
    if not media_path.exists():
        console.print(f"[bold red]Error:[/bold red] Media directory '{media_dir}' does not exist.")
        console.print(
            "Run [bold cyan]gemini-search samples[/bold cyan] first to create sample assets."
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold blue]Starting indexing with Gemini Embedding 2[/bold blue]\n"
            f"Media Directory: {media_path.resolve()}\n"
            f"Dimensions: {settings.embedding_dimensions} (MRL)\n"
            f"Vector DB: ChromaDB ({settings.chroma_persist_directory})"
        )
    )

    try:
        count = index_directory(media_dir=media_path, rate_limit_delay=delay)
        console.print(f"[bold green]Successfully indexed {count} items into ChromaDB![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed during indexing:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Text prompt to search across indexed media")],
    kind: Annotated[
        str | None,
        typer.Option("--kind", "-k", help="Filter results by modality: image, audio, pdf, video"),
    ] = None,
    prefix: Annotated[
        bool,
        typer.Option("--prefix/--no-prefix", help="Whether to apply the official task prefix"),
    ] = True,
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-n", help="Number of results to retrieve"),
    ] = 3,
    plot: Annotated[
        bool,
        typer.Option(
            "--plot/--no-plot",
            help="Display visual image grid using matplotlib if images are found",
        ),
    ] = False,
) -> None:
    """Search indexed media using natural language text."""
    console.print(
        f"[cyan]Searching for:[/cyan] [bold]'{query}'[/bold] (prefix={prefix}, kind={kind})"
    )
    try:
        results = search_by_text(
            query_text=query,
            kind=kind,
            use_prefix=prefix,
            n_results=top_k,
        )
        render_results_table(results, title=f"Results for '{query}'")

        if plot and any(r.get("kind") == "image" for r in results):
            plot_image_results(results)
    except Exception as e:
        console.print(f"[bold red]Search error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("search-image")
def search_image_cmd(
    image_path: Annotated[str, typer.Argument(help="Path to the query image (e.g. query.jpg)")],
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-n", help="Number of results to retrieve"),
    ] = 3,
    plot: Annotated[
        bool,
        typer.Option("--plot/--no-plot", help="Display visual image grid using matplotlib"),
    ] = False,
) -> None:
    """Search image database using an image as query (Visual Search)."""
    img_p = Path(image_path)
    if not img_p.exists():
        console.print(f"[bold red]Error:[/bold red] Query image '{image_path}' not found.")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Performing visual search with:[/cyan] [bold]{img_p.name}[/bold]")
    try:
        results = search_by_image(
            image_path=img_p,
            kind="image",
            n_results=top_k,
        )
        render_results_table(results, title=f"Visual Search: {img_p.name}")

        if plot:
            plot_image_results(results)
    except Exception as e:
        console.print(f"[bold red]Search error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Display information about the ChromaDB vector database and environment."""
    store = MultimodalVectorStore()
    total = store.count()

    info_text = (
        f"[bold]Model:[/bold] {settings.gemini_embedding_model}\n"
        f"[bold]Dimensions (MRL):[/bold] {settings.embedding_dimensions}\n"
        f"[bold]Persist Directory:[/bold] {settings.chroma_persist_directory}\n"
        f"[bold]Collection:[/bold] {settings.collection_name}\n"
        f"[bold]Indexed Items:[/bold] {total}\n"
        f"[bold]Distance Metric:[/bold] Cosine (hnsw:space = cosine)"
    )
    console.print(Panel(info_text, title="📊 Gemini Embedding 2 Database Status", expand=False))


@app.command()
def clear() -> None:
    """Clear all indexed vectors from the database."""
    store = MultimodalVectorStore()
    store.clear()
    console.print("[bold yellow]✓ ChromaDB collection cleared and reset.[/bold yellow]")


if __name__ == "__main__":
    app()
