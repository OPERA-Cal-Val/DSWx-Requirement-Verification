from pathlib import Path

import click
import papermill as pm
from dswx_verification import get_localized_validation_table
from tqdm import tqdm


repo_dir = Path(__file__).parent.resolve()
SITE_NOTEBOOKS_RELATIVE_PATHS = [repo_dir / 'notebooks/1_Verification_Metrics_at_Validation_Site.ipynb',
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
@click.option('--sites',
              default=None,
              type=str,
              multiple=True,
              required=False,
              help=('Site name to pass as parameter e.g. 3_10 to verification; used mainly for testing CLI; '
                    'Will still aggregate all data that has been serialized')
              )
@click.option('--sites-to-exclude',
              default=None,
              type=str,
              multiple=True,
              required=False,
              help=('Sites to exclude; for experimental validation')
              )
@click.command()
def main(yaml_config: str,
         output_notebooks_dir: str = None,
         sites: list = None,
         sites_to_exclude: list = None):
    p = output_notebooks_dir or 'out_notebooks'
    ipynb_dir = Path(p)
    ipynb_dir.mkdir(exist_ok=True, parents=True)

    df_val = get_localized_validation_table()
    site_names = df_val.site_name.tolist()
    if sites:
        valid_sites = [site in site_names for site in sites]
        if not all(valid_sites):
            raise ValueError('Site Names must be in the existing validation table e.g. 3_10')
        site_names = sites

    if sites_to_exclude:
        valid_sites = [site in site_names for site in sites_to_exclude]
        if not all(valid_sites):
            raise ValueError('Site Names to exclude must be in the existing validation table e.g. 3_10')
        site_names = [site for site in site_names if site not in sites_to_exclude]

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
