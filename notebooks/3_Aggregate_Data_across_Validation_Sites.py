# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: dswx_val
#     language: python
#     name: dswx_val
# ---

# %% jupyter={"source_hidden": true}
from dswx_verification.data_models import VerificationParameters
from pathlib import Path
import json
import pandas as pd
import numpy as np
import yaml

# %% [markdown]
# # Parameters

# %%
yaml_file = 'verification_parameters.yml'

# %% [markdown]
# Read parameters

# %%
verif_params = VerificationParameters.from_yaml(yaml_file)
verif_params

# %% [markdown]
# # Presentation Dir

# %%
with open(yaml_file) as f:
    presentation_params = yaml.safe_load(f)['presentation_parameters']

# %%
presentation_dir =  Path(presentation_params['presentation_dir']) 
presentation_dir.exists(), presentation_dir


# %% [markdown]
# # Read Processed Data

# %%
def get_site_ids_processed(data_dir: str | Path) -> list:
    data_dir = Path(data_dir)
    dswx_verification_paths = list(data_dir.glob('*/'))
    site_names_processed = [path.name for path in dswx_verification_paths]
    # Remove mac paths
    site_names_processed = list(filter(lambda path: '.' != path[0], site_names_processed))
    return site_names_processed



# %%
sites_processed = get_site_ids_processed(verif_params.data_dir)
sites_processed[:3]


# %%
def read_trial_data_from_site(site_name):
    data_dir = Path(verif_params.data_dir)
    json_path = data_dir / site_name / 'trial_stats.json'
    data = json.load(open(json_path))
    return data

processed_data = list(map(read_trial_data_from_site, sites_processed))


# %% [markdown]
# # Tables
#
# ## Mean accuracy metrics
#
# Means of all the accuracy statistics

# %%
def format_label(label):
    """get first letter for acronym"""
    if label in ['All', 'OSW $+$ PSW']:
        return label
    else:
        return ''.join(x[0]for x in label.split(' '))

trial_data = [{'Class': class_label.replace('_', ' '),
               'Metric': metric_label.replace('_per_class', '').capitalize().replace('Acc', 'Binary Acc.').replace('_', ' ') + ' $(\%)$',
               'Mean': np.mean([item.get(f'{metric_label}.{class_label}.mean', np.nan) * 100 for item in processed_data]),
               'Median': np.mean([item.get(f'{metric_label}.{class_label}.median', np.nan) * 100 for item in processed_data]),
               'St. Dev': np.mean([item.get(f'{metric_label}.{class_label}.std', np.nan) * 100 for item in processed_data])
              }
              for class_label in ['All', 'Open_Surface_Water', 'Partial_Surface_Water']
              for metric_label in ['total_accuracy', 'recall', 'precision', 'f1_per_class', 'acc_per_class', 'binary_water_acc']
             ]
_ = [item.update({'Class': 'OSW $+$ PSW', 'Metric': 'Binary Acc. $(\%)$'}) for item in trial_data if 'Binary water acc' in item['Metric']]
df_acc = pd.DataFrame(trial_data)
df_acc = df_acc.dropna(axis=0)
df_acc = df_acc.round(2)
df_acc = df_acc.sort_values(by=['Class', 'Metric']).reset_index(drop=True)
df_acc['Class'] = df_acc['Class'].map(format_label)
df_acc = df_acc.set_index(['Class', 'Metric'])
df_acc

# %% [markdown]
# Count of requirements passed

# %%
latex = df_acc.style.to_latex(multirow_align='t', hrules=True)
with open(presentation_dir / 'total_accuracy_for_all_validation.tex', 'w') as f:
    f.write(latex)

# %% [markdown]
# ## Count of requirements passed

# %%
df_proc = pd.DataFrame(processed_data)
df_proc.head()

# %%
n_osw_passes = df_proc.osw_requirement.sum()
n_pws_passes = df_proc.psw_requirement.sum()
n_both_pass = (df_proc.osw_requirement & df_proc.psw_requirement).sum()
n_pws_passes, n_osw_passes, n_both_pass

# %%
n_osw_fails = (~df_proc.osw_requirement).sum()
n_pws_fails = (~df_proc.psw_requirement).sum()
n_both_fail = (~df_proc.osw_requirement & ~df_proc.psw_requirement).sum()
n_osw_fails, n_pws_fails, n_both_fail

# %%
df_passes = pd.DataFrame([{'Class': 'Open Surface Water (OSW)',
                          'Pass': n_osw_passes,
                          'Not Pass': n_osw_fails},
                         {'Class': 'Partial Surface Water (PSW)',
                          'Pass': n_pws_passes,
                          'Not Pass': n_pws_fails},
                         {'Class': 'Both (OSW + PSW)',
                          'Pass': n_both_pass,
                          'Not Pass': n_both_fail}])
df_passes = df_passes.set_index('Class')
df_passes

# %%
latex = df_passes.style.to_latex(multirow_align='t', hrules=True)
with open(presentation_dir / 'total_passes.tex', 'w') as f:
    f.write(latex)
