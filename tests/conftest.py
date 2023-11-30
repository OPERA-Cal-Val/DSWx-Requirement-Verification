from pathlib import Path

import pytest
from jinja2 import Template


TEST_DIR = Path(__file__).parent.resolve()


@pytest.fixture(scope='session')
def verify_all_script_path():
    return TEST_DIR.parent / 'verify_all_dswx.py'


@pytest.fixture(scope='function')
def yaml_config_path():
    """Generates a yml config with absolute path to DB directory; returns path of yml file"""
    with open(TEST_DIR / 'data' / 'verification_parameters_template.yml', 'r') as file:
        template = Template(file.read())
    yml_config = template.render(DB=str(TEST_DIR / 'data' / 'DB'))
    out_path = TEST_DIR / 'data' / 'config.yml'
    with open(out_path, 'w') as file:
        file.write(yml_config)
    return out_path
