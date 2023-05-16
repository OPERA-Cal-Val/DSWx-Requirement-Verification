from pathlib import Path

import click
import papermill as pm
from dswx_verification import get_localized_validation_table
from tqdm import tqdm


repo_dir = Path(__file__).parent.resolve()
SITE_NOTEBOOKS_RELATIVE_PATHS = [ repo_dir / 'notebooks/1_Verification_Metrics_at_Validation_Site.ipynb',
                                  repo_dir / 'notebooks/2_Summarize_and_Visualize_Data_at_Validation_Site.ipynb',
                                 ]

@click.option('--yaml_config',
              required=True,
              type=str,
              help='Relative path to yaml file')
@click.option('--output_notebooks_dir',
              default=None,
              type=str,
              required=False,
              help='Where output notebooks will be saved')
@click.command()
def main(yaml_config, output_notebooks_dir: str):
    p = output_notebooks_dir or 'out_notebooks'
    ipynb_dir = Path(p)
    ipynb_dir.mkdir(exist_ok=True, parents=True)

    df_val = get_localized_validation_table()
    site_names = df_val.site_name.tolist()

    for in_nb in SITE_NOTEBOOKS_RELATIVE_PATHS:
        print(f'Using the {in_nb.name} template')
        for site_name in tqdm(site_names):
            out_nb = ipynb_dir / f'{site_name}.ipynb'

            print(f'Site Name: {site_name}')
            pm.execute_notebook(in_nb,
                                output_path=out_nb,
                                parameters=dict(site_name=site_name,
                                                yaml_file=yaml_config)
                                )
    in_nb = repo_dir / 'notebooks' / '3_Aggregate_Data_across_Validation_Sites.ipynb'
    out_nb = ipynb_dir / in_nb.name
    print(f'Using the {in_nb.name} template')
    pm.execute_notebook(in_nb,
                        output_path=out_nb,
                        parameters=dict(yaml_file=yaml_config))

    in_nb = repo_dir / 'notebooks' / '4_Generate_Beamer_Slides.ipynb'
    out_nb = ipynb_dir / in_nb.name
    print(f'Using the {in_nb.name} template')
    pm.execute_notebook(in_nb,
                        output_path=out_nb,
                        parameters=dict(yaml_file=yaml_config))


if __name__ == '__main__':
    main()
