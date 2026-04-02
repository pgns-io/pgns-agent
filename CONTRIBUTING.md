# Contributing

This repository is synced from a private monorepo, but contributions are actively encouraged. The agent library ecosystem is the primary surface for community involvement.

## What PRs are welcome on

- `examples/` — new examples, usage patterns, integrations with other tools
- `docs/` — documentation fixes, additions, tutorials
- `README.md` — typo fixes, improved descriptions, usage examples
- `tests/` — additional test coverage
- New adapter integrations and community extensions

Community-contributed adapters and examples are especially valuable. If an agent framework is missing adapter support, a PR adding it is the fastest path to getting it shipped.

## What to open an issue for

Core library files (`pgns_agent/`, `pyproject.toml`, etc.) are synced from an upstream monorepo. PRs that modify these files can't be merged directly because the next sync would overwrite them.

If a change is needed in core code, open an issue describing the problem or desired behavior. The maintainers will apply the fix upstream, and it will ship in the next release.

## How accepted PRs work

When a PR lands on a safe zone listed above, a maintainer reviews it, applies the change in the monorepo, and the next sync publishes it to this repo. The PR itself may be closed rather than merged — the change arrives via the sync pipeline instead.

## Development

Clone the repo and install dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

By contributing, contributions are licensed under the same terms as this project (see [LICENSE](LICENSE)).
