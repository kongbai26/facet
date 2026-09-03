"""Command-line entry points for Facet."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")


def _add_server_options(parser: argparse.ArgumentParser) -> None:
    """Add the shared bind and logging options for server commands."""
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Bind port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="info",
        help="Server log level (default: info)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the Facet command parser."""
    parser = argparse.ArgumentParser(
        prog="facet",
        description="Facet — a local knowledge base powered by RAG.",
    )
    parser.add_argument("--version", action="version", version="Facet 0.1.0")

    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    serve_parser = commands.add_parser(
        "serve",
        help="Start Facet for normal use.",
        description="Start Facet without auto-reload.",
    )
    _add_server_options(serve_parser)

    dev_parser = commands.add_parser(
        "dev",
        help="Start Facet with automatic reload for development.",
        description="Start Facet with automatic reload for development.",
    )
    _add_server_options(dev_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Facet command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.command == "dev",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the package entry point
    raise SystemExit(main())
