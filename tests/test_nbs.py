import sys

from click.testing import CliRunner


def test_notebook_integration(verify_all_script_path,
                              yaml_config_path):
    # Add manually the verify all script to python path; would be more work to make an actual CLI and would defeat
    # interactivity thrust of this repository
    repo_dir = str(verify_all_script_path.parent)
    sys.path.append(repo_dir)
    from verify_all_dswx import main

    runner = CliRunner()
    result = runner.invoke(main, ['--yaml_config', str(yaml_config_path),
                                  '--output_notebooks_dir', None,
                                  '--sites', '3_10'])
    # This ensures we can more easily see output of CLI
    print(result.stdout)
    print(result.exception)
    assert result.exit_code == 0
