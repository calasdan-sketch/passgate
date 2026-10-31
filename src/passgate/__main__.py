"""Allow ``python -m passgate``."""
import sys

from .cli import main

sys.exit(main())
