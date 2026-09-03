# Contributing to Facet

Thank you for helping improve Facet. Keep changes focused, preserve tenant and
knowledge-base boundaries, and never commit local configuration or runtime
data.

## Development checks

```bash
python -m pip install -e ".[dev]"
python -m ruff check app scripts

cd web
npm ci
npm run lint
npm run build
npm audit --omit=dev --audit-level=high
```

Routing changes must keep route selection, evidence resolution, and answer
policy separate. Evaluation scripts must put every mutable store in a
temporary work directory.

Security-sensitive reports should follow [SECURITY.md](./SECURITY.md).
