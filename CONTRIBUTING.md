# Contributing

Thanks for contributing to Plant Watering System.

## Development Setup

1. Fork and clone the repository.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Typical Workflow

1. Create a feature branch.
2. Make focused changes.
3. Run checks locally:

```bash
python -m compileall src app
python src/data/preprocess.py
python src/models/compare_models.py
```

4. Update docs for any user-visible behavior changes.
5. Open a PR with:
   - What changed
   - Why it changed
   - Validation steps

## Code Guidelines

- Prefer small, readable functions.
- Keep config in `src/utils/config.py`.
- Keep data docs updated in `data/README.md`.
- Avoid committing large generated artifacts unless they are required for reproducibility/demo.

## Reporting Issues

Please include:
- OS + Python version
- Command used
- Full traceback/error output
- Steps to reproduce
