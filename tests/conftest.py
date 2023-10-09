from pathlib import Path

import pytest

test_dir = Path(__file__).parents[0]
TEST_DIR = test_dir.resolve()


@pytest.fixture(scope='session')
def verify_all_script_path():
    return TEST_DIR.parent / 'verify_all_dswx.py'
