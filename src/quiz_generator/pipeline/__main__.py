"""Entry point for `python -m quiz_generator.pipeline`."""

from quiz_generator.pipeline.cli import main
from quiz_generator.shared.logging_setup import setup_logging

if __name__ == "__main__":
    setup_logging()
    main()
