"""
main.py

CLI entry point for the Autonomous PM Engine.

Usage:
    python main.py --input-dir sample_data/ --product-name "MyProduct" \\
                   --product-context "B2B SaaS project management tool"

    python main.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config.settings import get_settings
from src.orchestration.workflow import run_pipeline

console = Console()


def _configure_logging() -> None:
    cfg = get_settings()
    log_dir = Path(cfg.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=cfg.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        cfg.log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous PM Engine — Generate PRDs from raw customer feedback.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input-dir sample_data/
  python main.py --input-dir ./data --product-name "Acme Search" \\
                 --product-context "Enterprise search tool for legal teams"
        """,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="sample_data/",
        help="Directory containing raw feedback documents (default: sample_data/)",
    )
    parser.add_argument(
        "--product-name",
        type=str,
        default="Product",
        help='Product name for the PRD title (default: "Product")',
    )
    parser.add_argument(
        "--product-context",
        type=str,
        default="A software product used by business customers.",
        help="One-sentence description of the product domain",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from .env settings",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = _parse_args()

    # Override output dir if specified on CLI
    if args.output_dir:
        import os
        os.environ["OUTPUT_DIR"] = args.output_dir
        # Bust the settings cache
        get_settings.cache_clear()

    cfg = get_settings()

    # ── Welcome banner ─────────────────────────────────────────────────────
    console.print(
        Panel.fit(
            "[bold cyan]Autonomous PM Engine[/bold cyan]\n"
            "[dim]Multi-agent PRD generation from customer feedback[/dim]",
            border_style="cyan",
        )
    )

    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_row("[bold]Product[/bold]", args.product_name)
    config_table.add_row("[bold]Input[/bold]", args.input_dir)
    config_table.add_row("[bold]Output[/bold]", cfg.output_dir)
    config_table.add_row("[bold]LLM[/bold]", cfg.openai_model)
    config_table.add_row("[bold]Embeddings[/bold]", cfg.embedding_model)
    console.print(config_table)
    console.print()

    # ── Validate input dir ──────────────────────────────────────────────────
    input_path = Path(args.input_dir)
    if not input_path.exists():
        console.print(f"[red]ERROR:[/red] Input directory not found: {input_path}")
        sys.exit(1)

    # ── Run pipeline ────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Running pipeline...", total=None)
        try:
            final_state = run_pipeline(
                input_dir=args.input_dir,
                product_name=args.product_name,
                product_context=args.product_context,
            )
            progress.update(task, description="[green]Pipeline complete")
        except Exception as e:
            progress.update(task, description="[red]Pipeline failed")
            console.print(f"\n[red]FATAL ERROR:[/red] {e}")
            logger.exception("Pipeline crashed")
            sys.exit(1)

    # ── Summary ─────────────────────────────────────────────────────────────
    console.print()
    if final_state.get("completed"):
        results_table = Table(title="Output Files", show_header=True, header_style="bold green")
        results_table.add_column("Type", style="bold")
        results_table.add_column("Path")

        results_table.add_row("PRD", final_state.get("final_prd_path", "N/A"))
        results_table.add_row("Roadmap", final_state.get("final_roadmap_path", "N/A"))
        results_table.add_row("Priority Matrix", final_state.get("final_matrix_path", "N/A"))

        console.print(results_table)

        stats_table = Table(title="Pipeline Stats", show_header=True, header_style="bold blue")
        stats_table.add_column("Metric")
        stats_table.add_column("Value", justify="right")
        stats_table.add_row("Raw documents", str(final_state.get("raw_document_count", 0)))
        stats_table.add_row("Semantic chunks", str(final_state.get("chunk_count", 0)))
        stats_table.add_row("Pain points extracted", str(final_state.get("pain_point_count", 0)))
        stats_table.add_row("Critique rounds", str(final_state.get("critique_rounds_completed", 0)))
        history = final_state.get("critique_history", [])
        if history:
            stats_table.add_row("Final quality score", f"{history[-1]['score']:.1f}/10")
        console.print(stats_table)

    if final_state.get("errors"):
        console.print("\n[yellow]Warnings:[/yellow]")
        for err in final_state["errors"]:
            console.print(f"  [yellow]-[/yellow] {err}")


if __name__ == "__main__":
    main()
