"""Enable `python -m autobench.noise_kernel ...` (delegates to the CLI)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
