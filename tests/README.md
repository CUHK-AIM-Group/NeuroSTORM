# NeuroSTORM Test Suite

This directory contains unit tests and integration tests for the NeuroSTORM platform.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── test_dual_format.py         # Tests for PT/H5 dual format support
├── test_atlas_masking.py       # Tests for atlas masking functionality
├── test_model_loading.py       # Tests for model loading and initialization
└── README.md                   # This file
```

## Running Tests

### Run all tests

```bash
pytest tests/
```

### Run specific test file

```bash
pytest tests/test_model_loading.py
```

### Run with coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Run only unit tests

```bash
pytest tests/ -m unit
```

### Run only integration tests

```bash
pytest tests/ -m integration
```

### Skip slow tests

```bash
pytest tests/ -m "not slow"
```

## Test Markers

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, multiple components)
- `@pytest.mark.slow` - Slow tests (can be skipped for quick checks)
- `@pytest.mark.gpu` - Tests that require GPU

## Continuous Integration

Tests are automatically run on every push and pull request via GitHub Actions.

See [.github/workflows/ci.yml](../.github/workflows/ci.yml) for CI configuration.

### CI Jobs

1. **test** - Run all tests on Python 3.9, 3.10, 3.11
2. **lint** - Check code style with flake8, black, isort
3. **model-tests** - Test model imports and initialization

## Writing Tests

### Test File Naming

- Test files must start with `test_`
- Test functions must start with `test_`
- Test classes must start with `Test`

### Example Test

```python
import pytest
import torch

class TestMyModel:
    """Test MyModel functionality."""

    @pytest.mark.unit
    def test_model_initialization(self):
        """Test model can be initialized."""
        from models.mymodel import MyModel
        
        model = MyModel(num_classes=2)
        assert model is not None

    @pytest.mark.unit
    def test_forward_pass(self):
        """Test forward pass."""
        from models.mymodel import MyModel
        
        model = MyModel(num_classes=2)
        x = torch.randn(2, 200, 200)
        
        output = model(x)
        assert output.shape == (2, 2)
```

## Test Coverage

Current test coverage can be viewed in the CI pipeline or by running:

```bash
pytest tests/ --cov=. --cov-report=term
```

## Adding New Tests

When adding new features, please add corresponding tests:

1. Create a new test file: `tests/test_<feature>.py`
2. Write test functions with descriptive names
3. Use appropriate markers (`@pytest.mark.unit`, etc.)
4. Run tests locally before pushing
5. Ensure CI passes

## Dependencies

Test dependencies are automatically installed in CI. For local testing:

```bash
pip install pytest pytest-cov pytest-xdist
```

## Troubleshooting

### Tests fail locally but pass in CI

- Check Python version (CI tests on 3.9, 3.10, 3.11)
- Check installed package versions
- Clear pytest cache: `pytest --cache-clear`

### Import errors

- Ensure you're in the project root directory
- Check that all dependencies are installed
- Verify PYTHONPATH includes project root

### GPU tests fail

- GPU tests are marked with `@pytest.mark.gpu`
- Skip GPU tests on CPU-only machines: `pytest -m "not gpu"`

---

**Last Updated:** 2026-05-08  
**Maintained by:** NeuroSTORM Team
