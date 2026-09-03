"""Allow ``python -m app`` to use the Facet command-line interface."""

from app.cli import main


raise SystemExit(main())
