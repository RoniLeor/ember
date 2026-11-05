"""Support python -m ember as well as the installed ember command."""

from .cli import main

raise SystemExit(main())
