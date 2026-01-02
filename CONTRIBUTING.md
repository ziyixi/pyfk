# Contributing to PyFK

Thank you for your interest in contributing to PyFK! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ziyixi/pyfk.git
   cd pyfk
   ```

2. **Install dependencies:**
   ```bash
   # Install Poetry if you haven't already
   pip install poetry

   # Install development dependencies
   poetry install --with dev,docs
   ```

3. **Install MPI (optional, for MPI support):**
   ```bash
   # Ubuntu/Debian
   sudo apt install openmpi-bin libopenmpi-dev

   # macOS
   brew install open-mpi

   # Install with MPI support
   PYFK_USE_MPI=1 pip install ".[mpi]"
   ```

## Running Tests

```bash
# Run all tests
pytest --pyargs pyfk

# Run with coverage
coverage run --source=pyfk -m pytest --pyargs pyfk
coverage report

# Run MPI tests
mpirun -np 3 pytest --with-mpi --pyargs pyfk

# Run numerical stability tests (requires hypothesis)
pytest pyfk/tests/test_numerical_stability.py -v
```

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check linting
ruff check pyfk

# Auto-fix linting issues
ruff check --fix pyfk

# Check formatting
ruff format --check pyfk

# Apply formatting
ruff format pyfk
```

## Pull Request Process

1. **Fork and create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and ensure:**
   - All tests pass
   - Code follows the project style
   - Documentation is updated if needed
   - Commit messages are clear and descriptive

3. **Submit a pull request:**
   - Fill out the PR template
   - Link any related issues
   - Wait for CI checks to pass

## Reporting Issues

When reporting bugs, please include:
- Python version
- PyFK version
- Operating system
- Full error traceback
- Minimal reproducible example

## Code of Conduct

Please be respectful and considerate in all interactions. We aim to maintain a welcoming and inclusive community.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
