"""Run a configured collection, replay, deployment, or serving workflow."""
from .core.config import run_cli


def main(argv=None):
    run_cli(__doc__, argv)


if __name__ == "__main__":
    main()
