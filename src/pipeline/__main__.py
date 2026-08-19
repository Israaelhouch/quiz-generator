"""Entry point for `python -m src.pipeline`."""

from src.pipeline.cli import main
from src.shared.logging_setup import setup_logging

if __name__ == "__main__":
    setup_logging()
    main()
