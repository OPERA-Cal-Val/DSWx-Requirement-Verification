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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from dswx_verification.data_models import VerificationParameters
from dswx_verification import get_validation_metadata_by_site_name
import rasterio
import geopandas as gpd
from matplotlib.colors import ListedColormap
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import json
from rasterio.plot import show
from matplotlib import colors
import yaml

# %% [markdown]
# # Parameters

# %% tags=["parameters"]
site_name = '3_10'
yaml_file = 'verification_parameters.yml'

# %% [markdown]
# # Load Ids and Set up Directories

# %%
verif_params = VerificationParameters.from_yaml(yaml_file)
verif_params

# %%
df_site_meta = get_validation_metadata_by_site_name(site_name)
df_site_meta

# %%
dswx_id = df_site_meta['dswx_id'][0]
planet_id = df_site_meta['planet_id'][0]

# %%
all_data_dir = Path(verif_params.data_dir)
assert all_data_dir.exists()

# %%
site_dir = all_data_dir / site_name
assert site_dir.exists()

# %% [markdown]
# # Presentation

# %%
with open(yaml_file) as f:
    presentation_params = yaml.safe_load(f)['presentation_parameters']

# %%
presentation_dir =  Path(presentation_params['presentation_dir']) / site_name
presentation_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Load Data
#
# ## Rasters

# %%
raster_path_names = [f'validation-dataset__{site_name}.tif',
                     f'dswx-mask__{site_name}.tif',
                     f'validation-dataset-rprj__{site_name}.tif',
                     f'cropped-dswx__{dswx_id}.tif',
                     f'dswx__{dswx_id}.tif'
                    ]
raster_paths_dict = {name.split('__')[0]: site_dir / name for name in raster_path_names}
[path.exists() for path in raster_paths_dict.values()]

# %%
with rasterio.open(raster_paths_dict['validation-dataset']) as ds:
    p_val = ds.profile
    X_val = ds.read(1)

with rasterio.open(raster_paths_dict['validation-dataset-rprj']) as ds:
    p_val_r = ds.profile
    X_val_r = ds.read(1)
    
with rasterio.open(raster_paths_dict['cropped-dswx']) as ds:
    p_dswx_c = ds.profile
    X_dswx_c = ds.read(1)
    
with rasterio.open(raster_paths_dict['dswx']) as ds:
    p_dswx = ds.profile
    X_dswx = ds.read(1)
    colormap = ds.colormap(1)
    
    
with rasterio.open(raster_paths_dict['dswx-mask']) as ds:
    dswx_mask = ds.read(1).astype(bool)


# %% [markdown]
# # Plots at Validation Site

# %%
cmap = ListedColormap([np.array(colormap[key]) / 255 for key in range(256)])

# %%
fontSize=7
fontSizeTitles=8
fig, ax = plt.subplots(1, 3, dpi=250, figsize=(20, 10))

dswx_im_data = show(X_dswx, cmap=cmap, transform=p_dswx['transform'], vmin=0, vmax=255, interpolation='none', ax=ax[0])
df_site_meta.to_crs(p_dswx['crs']).boundary.plot(ax=ax[0], color='black')
ax[0].set_title('Full DSWx scene with val bbox',fontsize=fontSize)
ax[0].set_xlabel('UTM easting (meters)',fontsize=fontSizeTitles)
ax[0].set_ylabel('UTM northing (meters)',fontsize=fontSizeTitles)
ax[0].ticklabel_format(axis='both', style='scientific',scilimits=(0,0),useOffset=False,useMathText=True)
ax[0].tick_params(axis='both', which='major', labelsize=fontSize)
ax[0].yaxis.get_offset_text().set_fontsize(fontSize)
ax[0].xaxis.get_offset_text().set_fontsize(fontSize)

show(X_dswx, cmap=cmap, transform=p_dswx['transform'], vmin=0,vmax=255, interpolation='none', ax=ax[1])
val_bounds_dswx = df_site_meta.to_crs(p_dswx['crs']).total_bounds
ax[1].set_xlim(val_bounds_dswx[0], val_bounds_dswx[2])
ax[1].set_ylim(val_bounds_dswx[1], val_bounds_dswx[3])
ax[1].set_title('DSWx Subset Area',fontsize=fontSizeTitles)
ax[1].set_xlabel('UTM easting (meters)',fontsize=fontSize)
ax[1].ticklabel_format(axis='both', style='scientific',scilimits=(0,0),useOffset=False,useMathText=True)
ax[1].tick_params(axis='both', which='major', labelsize=fontSize)
ax[1].yaxis.get_offset_text().set_fontsize(fontSize)
ax[1].xaxis.get_offset_text().set_fontsize(fontSize)

show(X_val, transform=p_val['transform'], ax=ax[2], interpolation='none', cmap=cmap, vmin=0, vmax=255)
ax[2].set_title('Validation Dataset',fontsize=fontSizeTitles)
ax[2].set_xlabel('UTM easting (meters)',fontsize=fontSize)
ax[2].ticklabel_format(axis='both', style='scientific',scilimits=(0,0),useOffset=False,useMathText=True)
ax[2].tick_params(axis='both', which='major', labelsize=fontSize)
ax[2].yaxis.get_offset_text().set_fontsize(fontSize)
ax[2].xaxis.get_offset_text().set_fontsize(fontSize)

im = dswx_im_data.get_images()[0]
bounds =  [0, 1, 2, 3, 
           251, 252, 253, #254
          ]
cbar=fig.colorbar(im, 
                  ax=ax, 
                  shrink=0.5, 
                  pad=0.05, 
                  boundaries=bounds, 
                  cmap=cmap, 
                  ticks=[0.5, 1.5, 2.5, 251.5, 252.5]) #, 253.5])

cbar.ax.tick_params(labelsize=8)
norm = colors.BoundaryNorm(bounds, cmap.N)
cbar.set_ticklabels(['Not Water', 
                     'Open Water',
                     'Partial Surface Water',
                     'Snow/Ice','Cloud/Cloud Shadow', 
                     #'Ocean Mask'
                    ],
                    fontsize=fontSize)   



#plt.tight_layout()
plt.savefig(presentation_dir / 'extent.png')

# %%
for plot_type in ['without_mask', 'with_mask']:
    fontSize=6
    fig, ax = plt.subplots(1, 2, dpi=150, figsize=(10, 5))
    im=ax[0].imshow(X_dswx_c, interpolation='none',cmap=cmap,vmin=0,vmax=255)

    out = show(X_dswx_c, cmap=cmap, transform=p_dswx_c['transform'], interpolation='none', ax=ax[0], vmin=0,vmax=255)
    im_dswx = out.get_images()[0]

    ax[0].set_title('DSWx-HLS (30 m)',fontsize=8)
    ax[0].set_xlabel('UTM easting (meters)',fontsize=fontSize)
    ax[0].set_ylabel('UTM northing (meters)',fontsize=fontSize)
    ax[0].ticklabel_format(axis='both', style='scientific',scilimits=(0,0),useOffset=False,useMathText=True)
    ax[0].tick_params(axis='both', which='major', labelsize=fontSize)
    ax[0].yaxis.get_offset_text().set_fontsize(fontSize)
    ax[0].xaxis.get_offset_text().set_fontsize(fontSize)

    show(X_val_r, cmap=cmap, transform=p_val_r['transform'], interpolation='none',ax=ax[1], vmin=0, vmax=255)
    ax[1].set_title('Validation Data (30 m)',fontsize=8)
    ax[1].set_xlabel('UTM easting (meters)',fontsize=fontSize)
    ax[1].set_ylabel('UTM northing (meters)',fontsize=fontSize)

    ax[1].ticklabel_format(axis='both', style='scientific',scilimits=(0,0),useOffset=False,useMathText=True)
    ax[1].tick_params(axis='both', which='major', labelsize=fontSize)
    ax[1].yaxis.get_offset_text().set_fontsize(fontSize)
    ax[1].xaxis.get_offset_text().set_fontsize(fontSize)

    display_mask = dswx_mask.astype(np.float32)
    display_mask[~dswx_mask] = np.nan


    if plot_type == 'with_mask':
        show(display_mask, 
             cmap='viridis', 
             transform=p_val_r['transform'], 
             interpolation='none',
             ax=ax[0], vmin=0, vmax=1, alpha=1)

        show(display_mask, 
             cmap='viridis', 
             transform=p_val_r['transform'], 
             interpolation='none',
             ax=ax[1], vmin=0, vmax=1, alpha=1)


    cbar.ax.tick_params(labelsize=8)
    bounds =  [0, 1, 2, 3, 
               251, 252, 253, # 254
              ]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cbar=fig.colorbar(im_dswx, 
                      ax=ax, shrink=0.5, pad=0.05, boundaries=bounds, cmap=cmap, 
                      ticks=[0.5, 1.5, 2.5, 
                             251.5, 252.5, # 253.5
                            ])
    cbar.set_ticklabels(['Not Water', 
                         'Open Water',
                         'Partial Surface Water',
                         'Snow/Ice',
                         'Cloud/Cloud Shadow', 
                         #'Ocean Mask'
                        ]
                        ,fontsize=fontSize)   

    cbar.ax.tick_params(labelsize=8)

    ### colorbar

    # source: https://stackoverflow.com/questions/39500265/manually-add-legend-items-python-matplotlib

    if plot_type == 'with_mask':
        from matplotlib.lines import Line2D
        from matplotlib import colors, colorbar, cm
        cNorm  = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap='viridis')
        legend_elements = [Line2D([0], 
                                  [0], 
                                  marker='s',
                                  color='w',
                                  label='No data/Masked',
                                  markerfacecolor=scalarMap.to_rgba(1),
                                  markeredgecolor='black',
                                  alpha=1,
                                  markersize=10)]
        ax[0].legend(handles=legend_elements, loc='upper left', fontsize=6,  framealpha=1, edgecolor='white')


    plt.savefig(presentation_dir / f'comparison_resample_{plot_type}.png')

# %% [markdown]
# # Samples of a fixed Trial

# %%
df_samples = gpd.read_file(site_dir / f'samples__{site_name}')

# %%
fig, ax = plt.subplots(dpi=300)

plot.show(X_val_r, transform=p_val_r['transform'], ax=ax, cmap=cmap, vmin=0, vmax=255, interpolation='none')
df_samples.plot(ax=ax, color='green', alpha=.85)

ax.set_xlabel('UTM easting (meters)',fontsize=fontSize)
ax.set_ylabel('UTM northing (meters)',fontsize=fontSize)

ax.ticklabel_format(axis='both', style='scientific',scilimits=(0,0),useOffset=False,useMathText=True)
ax.tick_params(axis='both', which='major', labelsize=fontSize)
ax.yaxis.get_offset_text().set_fontsize(fontSize)
ax.xaxis.get_offset_text().set_fontsize(fontSize)
    
plt.savefig(presentation_dir / f'sampling_of_trial_over_validation_site.png')

# %% [markdown]
# # Summarize Metrics at Validation Site
#
# This involves taking the metric json data and formatting it into a table (mostly by hand) so that it looks nice in a latex presentation.

# %%
metric_data  = json.loads(Path(site_dir / 'trial_stats.json').read_text())

# %% [markdown]
# ## Precision / Recall / Total Accuracy

# %%
dict_for_acc = {key: val for (key, val) in metric_data.items()
                if any(kw in key for kw in ['precision', 'recall', 'total_accuracy', 'f1', 'supp']) 
                and any(kw not in key for kw in ['Not_Water'])}
dict_for_acc

# %%
stat_data_0 = [{'Class': class_label.replace('_', ' '),
               'Metric': metric_label.replace('_per_class', '').capitalize().replace('_', ' ') + ' $(\%)$',
               'Mean': dict_for_acc.get(f'{metric_label}.{class_label}.mean', np.nan) * 100,
               'Median': dict_for_acc.get(f'{metric_label}.{class_label}.median', np.nan) * 100,
               'St. Dev': dict_for_acc.get(f'{metric_label}.{class_label}.std', np.nan) * 100
              }
              for class_label in ['All', 'Open_Surface_Water', 'Partial_Surface_Water']
              for metric_label in ['total_accuracy', 'recall', 'precision', 'f1_per_class']
             ]
stat_data_1 = [{'Class': class_label.replace('_', ' '),
               'Metric': metric_label.replace('_per_class', '').capitalize().replace('_', ' '),
               'Mean': dict_for_acc.get(f'{metric_label}.{class_label}.mean', np.nan) ,
               'Median': dict_for_acc.get(f'{metric_label}.{class_label}.median', np.nan),
               'St. Dev': dict_for_acc.get(f'{metric_label}.{class_label}.std', np.nan)
              }
              for class_label in ['All', 'Open_Surface_Water', 'Partial_Surface_Water']
              for metric_label in ['supp_per_class']
             ]

# More Formatting
df_acc = pd.DataFrame(stat_data_0 + stat_data_1)
df_acc = df_acc.dropna(axis=0)
df_acc = df_acc.round(2).astype(str)
df_acc = df_acc.sort_values(by=['Class', 'Metric']).reset_index(drop=True)
df_acc = df_acc.set_index(['Class', 'Metric'])
df_acc

# %%
latex = df_acc.style.to_latex(multirow_align='t', hrules=True)
with open(presentation_dir / 'accuracy.tex', 'w') as f:
    f.write(latex)

# %% [markdown]
# ## Confusion Matrix

# %%
dict_for_conf = {key: val for (key, val) in metric_data.items()
                if 'confusion' in key and 'median' not in key}
dict_for_conf

# %%
labels = ['Not_Water',  'Open_Surface_Water', 'Partial_Surface_Water']
conf_mean = [[dict_for_conf.get(f'confusion_matrix.{label_1}_OPERA_DSWx.{label_2}_OPERA_Validation.mean', 0)
                       for label_1 in labels] 
                      for label_2 in labels]
conf_mean

# %%
conf_std = [[dict_for_conf.get(f'confusion_matrix.{label_1}_OPERA_DSWx.{label_2}_OPERA_Validation.std', 0)
                       for label_1 in labels] 
                      for label_2 in labels]
conf_std

# %%
conf_data = [[f'{mu:1.2f} ({std:1.2f})' for (mu, std) in zip(mu_list, std_list)] for (mu_list, std_list) in zip(conf_mean, conf_std)]
conf_data

# %%
df_confusion = pd.DataFrame(conf_data,
                            index=['NW (DSWx)', 'OSW (DSWx)', 'PSW (DSWx)'],
                            columns=['NW (Val)', 'OSW (Val)', 'PSW (Val)']
                           )
df_confusion

# %%
latex = df_confusion.style.to_latex(multirow_align='t', hrules=True)
with open(presentation_dir / 'confusion.tex', 'w') as f:
    f.write(latex)

# %% [markdown]
# ## Requirement Table

# %%
osw_mu = metric_data['acc_per_class.Open_Surface_Water.mean']
osw_std = metric_data['acc_per_class.Open_Surface_Water.std']
osw_req = metric_data['osw_requirement']

psw_mu = metric_data['acc_per_class.Partial_Surface_Water.mean']
psw_std = metric_data['acc_per_class.Partial_Surface_Water.std']
psw_req = metric_data['psw_requirement']

bw_mu = metric_data['binary_water_acc.All.mean']
bw_std = metric_data['binary_water_acc.All.std']

# %%

df_requirement = pd.DataFrame([{'Class': 'PSW',
                                'OPERA Req.': psw_req,
                                'Accuracy ($\%$)': f'{psw_mu * 100:1.2f} ({psw_std * 100:1.2f})'},
                               {'Class': 'OSW',
                                'OPERA Req.': osw_req,
                                'Accuracy ($\%$)': f'{osw_mu * 100:1.2f} ({osw_std * 100:1.2f})'},
                               {'Class': 'Binary Water',
                                'OPERA Req.': 'N/A',
                                'Accuracy ($\%$)': f'{bw_mu * 100:1.2f} ({bw_std * 100:1.2f})'}
                              ])

def labeler(val):
    if val:
        if val != 'N/A':
            return 'Passed'
        else:
            return val
    return 'Not Passed'

df_requirement['OPERA Req.'] = df_requirement['OPERA Req.'].map(labeler)
df_requirement

# %%
latex = df_requirement.style.hide(axis="index").to_latex(multirow_align='t', hrules=True)

with open(presentation_dir / 'requirements.tex', 'w') as f:
    f.write(latex)

# %% [markdown]
# ## Area of Class Labels

# %%
C_val = 9
C_dswx = 900
C_ha = 0.0001
area_data = [{'Type': 'Val',
              'Frame (Posting)': 'Validation (3 m)',
              'Class': 'OSW',
              'Area (ha)': (X_val == 1).sum() * C_ha * C_val,
              'Area ($\%$)': (X_val == 1).sum() / (X_val != 255).sum() * 100},
             {'Type': 'Val',
              'Frame (Posting)': 'Validation (3 m)',
              'Class': 'NW',
              'Area (ha)': (X_val == 0).sum() * C_ha * C_val,
              'Area ($\%$)': (X_val == 0).sum() / (X_val != 255).sum() * 100},
             {'Type': 'Val',
              'Frame (Posting)': 'DSWx (30 m)',
              'Class': 'OSW',
              'Area (ha)': (X_val_r[~dswx_mask] == 1).sum() * C_ha * C_dswx,
              'Area ($\%$)': (X_val_r[~dswx_mask] == 1).sum() / (~dswx_mask).sum() * 100},
             {'Type': 'Val',
              'Frame (Posting)': 'DSWx (30 m)',
              'Class': 'PSW',
              'Area (ha)': (X_val_r[~dswx_mask] == 2).sum() * C_ha * C_dswx,
              'Area ($\%$)': (X_val_r[~dswx_mask] == 2).sum() / (~dswx_mask).sum() * 100
             },
             {'Type': 'Val',
              'Frame (Posting)': 'DSWx (30 m)',
              'Class': 'NW',
              'Area (ha)': (X_val_r[~dswx_mask] == 0).sum() * C_ha * C_dswx,
              'Area ($\%$)': (X_val_r[~dswx_mask] == 0).sum() / (~dswx_mask).sum() * 100
             },
             {'Type': 'DSWx',
              'Frame (Posting)': 'DSWx (30 m)',
              'Class': 'OSW',
              'Area (ha)': (X_dswx_c[~dswx_mask] == 1).sum() * C_ha * C_dswx,
              'Area ($\%$)': (X_dswx_c[~dswx_mask] == 1).sum() / (~dswx_mask).sum() * 100},
             {'Type': 'DSWx',
              'Frame (Posting)': 'DSWx (30 m)',
              'Class': 'PSW',
              'Area (ha)': (X_dswx_c[~dswx_mask] == 2).sum() * C_ha * C_dswx,
              'Area ($\%$)': (X_dswx_c[~dswx_mask] == 2).sum() / (~dswx_mask).sum() * 100
             },
             {'Type': 'DSWx',
              'Frame (Posting)': 'DSWx (30 m)',
              'Class': 'NW',
              'Area (ha)': (X_dswx_c[~dswx_mask] == 0).sum() * C_ha * C_dswx,
              'Area ($\%$)': (X_dswx_c[~dswx_mask] == 0).sum() / (~dswx_mask).sum() * 100
             }
            ]
df_area = pd.DataFrame(area_data)
df_area

# %%
df_area_f = df_area.set_index(['Frame (Posting)', 'Type', 'Class'])
df_area_f['Area (ha)'] = df_area_f['Area (ha)'].map(lambda num: f'{num:1.2f}')
df_area_f['Area ($\%$)'] = df_area_f['Area ($\%$)'].map(lambda num: f'{num:1.2f}')
df_area_f

# %%
latex = df_area_f.style.to_latex(multirow_align='t', hrules=True)
with open(presentation_dir / 'areas.tex', 'w') as f:
    f.write(latex)

# %% [markdown]
# ## Ommision and Commision Error

# %%
data = {key: val for (key, val) in metric_data.items()
                if any(kw in key for kw in ['precision', 'recall'])}
data


# %%
def format_label(label):
    """get first letter for acronym"""
    return ''.join(x[0]for x in label.split('_'))

def format_metric(met):
    return 'om' if met == 'precision' else 'co' 

def get_metric(data: dict, met: str, class_label: str, stat: str):
    if stat == 'mean':
        return 100 - data.get(f'{met}.{class_label}.{stat}', np.nan) * 100
    else:
        return data.get(f'{met}.{class_label}.{stat}', np.nan) * 100

om_com_data = {f'{stat[0]}_{format_metric(met)}_{format_label(class_label)}': get_metric(data, met, class_label, stat)
              for class_label in ['Not_Water', 'Open_Surface_Water', 'Partial_Surface_Water']
              for stat in ['mean', 'std']
              for met in ['precision', 'recall']}
om_com_data = {key: f'{val:1.2f}' for key, val in om_com_data.items()}
om_com_data

# %%
table_data = {'Class': ['NW', 'OSW', 'PSW'],
               'Commission Error ($\%$)': [om_com_data[f'm_co_{l}'] + ' (' + om_com_data[f's_co_{l}'] + ')' for l in ['NW', 'OSW', 'PSW']],
               'Ommision Error ($\%$)': [om_com_data[f'm_om_{l}'] + ' (' + om_com_data[f's_co_{l}'] + ')' for l in ['NW', 'OSW', 'PSW']]
              }

df_om_co = pd.DataFrame(table_data)
df_om_co.set_index('Class', inplace=True)
df_om_co

# %%
latex = df_om_co.style.to_latex(multirow_align='t', hrules=True)
with open(presentation_dir / 'omission_comission.tex', 'w') as f:
    f.write(latex)

# %% [markdown]
# ## Ids for Validation Site

# %%
p_water_val = (X_val == 1).sum() / (X_val != 255).sum() * 100

def strata_lookup(p_water):
    if p_water <= 0:
        return 'No water (stratum 0)'
    if p_water <= .08:
        return 'Low water (stratum 1)'
    if p_water <= 2:
        return 'Medium Water (stratum 2)'
    return 'High water (stratum 3)'

stratum_string = strata_lookup(p_water_val) + f' with {p_water_val:1.2f}$\%$ water in validation scene'
stratum_string = '\\begin{itemize} \n\item ' + stratum_string + '\n\\end{itemize}'
stratum_string

# %%
latex_0 = "\\begin{verbatim}\n" + "Planet ID: " + planet_id + "\n\\end{verbatim}\n"
latex_1 = "\\begin{verbatim}\n" + "Site Name: " + site_name + "\n\\end{verbatim}\n"
latex_2 = "\\begin{verbatim}\n" + "DSWx-ID: " + dswx_id + "\n\\end{verbatim}\n"

latex = latex_0 + latex_1 + latex_2 + stratum_string
print(latex)

# %%
with open(presentation_dir / 'more_ids.tex', 'w') as f:
    f.write(latex)
