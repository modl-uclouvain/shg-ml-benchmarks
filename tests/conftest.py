from pathlib import Path

import pytest

from shg_ml_benchmarks.utils import _DATA_PATH_DFLT


@pytest.fixture
def dataset_available():
    if not Path(_DATA_PATH_DFLT).exists():
        return False
    return True
