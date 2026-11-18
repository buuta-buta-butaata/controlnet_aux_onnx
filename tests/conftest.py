import sys
import warnings
from os.path import abspath, dirname, join

target_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, target_path)

import pytest, random

@pytest.fixture(autouse=True)
def _fix_seed():
    random.seed(0)
    yield

