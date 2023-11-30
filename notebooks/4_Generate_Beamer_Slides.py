# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: dswx_val
#     language: python
#     name: dswx_val
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# I wanted to use jinja as [here](https://github.com/mondeja/latex-ji18n) and [here](https://tug.org/tug2019/slides/slides-ziegenhagen-python.pdf), but ended up using simple replace as I was having trouble figuring out the `jinja2.Environment` piece.

# %%
import yaml
from pathlib import Path
from dswx_verification import get_main_beamer_tex_template, render_latex_template, get_slide_tex_template
import jinja2

# %% [markdown]
# # Parameters

# %%
yaml_file = 'verification_parameters.yml'

# %% [markdown]
# # Load Parameters

# %%
with open(yaml_file) as f:
    presentation_params = yaml.safe_load(f)['presentation_parameters']

# %%
presentation_dir =  Path(presentation_params['presentation_dir']) 
presentation_dir.exists(), presentation_dir

# %% [markdown]
# # Read Templates

# %%
main_tex_tmpl = get_main_beamer_tex_template()
print(main_tex_tmpl)

# %%
slides_tmpl_tex = get_slide_tex_template()
print(slides_tmpl_tex)


# %% [markdown]
# # Generate Slides

# %%
def generate_one_slide(site_presentation_dir: Path) -> str:
    out_tex = render_latex_template(slides_tmpl_tex,
                                    # The site name has `_` and latex in normal text mode expects a `\_`
                                    variables=dict(siteName=site_presentation_dir.name.replace('_', '\_'), 
                                                   presentationSiteDir=str(site_presentation_dir)))
    return out_tex

def get_site_dirs_for_presentation(data_dir: str | Path) -> list:
    data_dir = Path(data_dir)
    dswx_verification_paths = list(data_dir.glob('*/'))
    sites_processed = [path for path in dswx_verification_paths]
    # Remove mac paths
    dirs = list(filter(lambda path: '.' != path.name[0], sites_processed))
    # Ensure directory
    dirs = list(filter(lambda path: path.is_dir(), dirs))
    # Relative path
    dirs = list(map(lambda path: path.relative_to(presentation_dir), dirs))
    return dirs



# %%
site_dirs = get_site_dirs_for_presentation(presentation_dir)

# %%
slides_list = list(map(generate_one_slide, site_dirs))
print(slides_list[0])

# %%
all_slides = '\n\n'.join(slides_list)
with open(presentation_dir / 'slides.tex', 'w') as f:
    f.write(all_slides) 

# %% [markdown]
# # Fill in Variables in Main

# %%
input_paths = dict(pathToPassexTex='total_accuracy_for_all_validation.tex',
                   pathToMeanStatsTex='total_passes.tex',
                   pathToSlides='slides.tex'
                   )
for _, val in input_paths.items():
    print((presentation_dir / val).exists())

# %%
n = str(len(site_dirs))
main_tex = render_latex_template(main_tex_tmpl, 
                                 input_paths=input_paths,
                                 variables=dict(numOfValidationSites=n))
print(main_tex)

# %%
with open(presentation_dir / 'main.tex', 'w') as f:
    f.write(main_tex) 
