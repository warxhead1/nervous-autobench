"""Enable `python -m autobench.svdag_beauty_kernel ...` (delegates to the CLI)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
