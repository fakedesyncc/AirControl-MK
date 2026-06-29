# Contributing

AirControl is an assistive application, so changes are reviewed with a strong
focus on safety, predictable input and clear diagnostics.

## Development Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m aircontrol
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m aircontrol
```

## Branches

- `main` is the stable branch.
- `develop` tracks the next integration state.
- `release/v*` branches prepare tagged releases.
- `feature/*` and `docs/*` branches should stay focused and small.
- `deploy` points to the revision intended for packaging/deployment.

## Checks Before a Pull Request

Run the fast checks before pushing:

```bash
python -m compileall aircontrol tests tools packaging/pyinstaller_hooks run_app.py
python -m unittest discover -s tests
python tools/check_tracked_sources.py
python -m aircontrol doctor --no-camera
python -m aircontrol selftest
```

If you touch the Go diagnostic helper:

```bash
gofmt -w cmd
go test ./cmd/aircontrol-helper
```

## Accessibility Rules

- Keep `View` and `Safe` modes non-invasive: no real mouse or keyboard events.
- Avoid gestures that can accidentally click, type or scroll without confirmation.
- Prefer dwell-click and scanning flows for users who cannot perform precise pinches.
- Diagnostics should explain blocked camera/input permissions without requiring a console.
- Do not commit personal datasets, support ZIP files, local configs or recordings.

## Pull Request Notes

Good PRs explain:

- what changed;
- which user scenario it improves;
- how accidental input is prevented;
- what checks were run.

