# Local Git hook for version bumps

To enable automatic version bumping from commit messages, run:

```bash
git config core.hooksPath .githooks
```

Commit message examples:

- `fix: improve badge repositioning`
  - default bump: `patch` (`0.0.1`)
- `feat: release: minor`
  - explicit `minor`
- `release: major`
  - explicit `major`
- `release: 1.4.2`
  - explicit semantic version
