import sys

from click.testing import CliRunner


def test_notebook_integration(verify_all_script_path):
    repo_dir = str(verify_all_script_path.parent)
    sys.path.append(repo_dir)
    from verify_all_dswx import main

    runner = CliRunner()
    result = runner.invoke(main, ['--yaml_config', f'{repo_dir}/verification_parameters.yml',
                                  '--output_notebooks_dir', None,
                                  '--sites', '3_10'])
    # This ensures we can more easily see output of CLI
    print(result.stdout)
    print(result.exception)
    assert result.exit_code == 0
