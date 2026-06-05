# Contributing to tmg-hmc

Thank you for your interest in contributing! This document outlines how to report issues, suggest features, and submit code changes.

## Reporting Bugs

Please open a [GitHub issue](https://github.com/erik-a-bensen/tmg_hmc/issues) and include:

- Your OS, Python version, and `tmg_hmc` version
- A minimal reproducible example
- The full error message/traceback

## Suggesting Features

Open a GitHub issue describing the feature and your use case. For anything that would change the public API or core algorithm, please open an issue for discussion before submitting a pull request.

## Contributing Code

### Setup

Fork the repository and clone your fork:
```bash
git clone https://github.com/<your-username>/tmg_hmc.git
cd tmg_hmc
```

Install in editable mode with all development dependencies:
```bash
pip install -e .[dev]
```

Install the pre-commit hooks:
```bash
pre-commit install
```

This will automatically run linting and formatting checks on every commit.

If you plan to modify the C++ extension files, also install `cppcheck` and `clang-format` via your system package manager:
- **Arch**: `sudo pacman -S cppcheck clang`
- **macOS**: `brew install cppcheck clang-format`
- **Ubuntu/Debian**: `sudo apt install cppcheck clang-format`
- **Windows**: `choco install cppcheck llvm` or `winget install Cppcheck.Cppcheck` and `winget install LLVM.LLVM`

### Making Changes

1. Create a branch from `main`:
   ```bash
   git checkout -b my-feature
   ```
2. Make your changes
3. Add or update tests as appropriate
4. Ensure all tests pass:
   ```bash
   pytest
   pytest -m "not gpu"  # if you don't have a GPU
   ```
5. Push your branch and open a pull request against `main`

### Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include a clear description of what changed and why
- All CI checks (tests + linting) must pass before merging
- For bug fixes, consider adding a test that would have caught the bug

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for Python linting and formatting, [clang-format](https://clang.llvm.org/docs/ClangFormat.html) for C++ formatting, and [cppcheck](https://cppcheck.sourceforge.io/) for C++ static analysis. All are run automatically via pre-commit.

To run all checks manually:
```bash
pre-commit run --all-files
```

To run ruff directly:
```bash
ruff check --fix src/
ruff format src/
```

## Questions

Feel free to open an issue if you have questions about the codebase or are unsure how to approach a contribution.