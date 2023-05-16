# DSWx-Requirement-Verification

This respository contains the workflow used to verify the DSWx suite. It is currently focused on DSWx-HLS, but will be updated along with the subsequent releases of DSWx products. At a high leve, the repository balances reproducibility and clarity. Here, "reproducibility" encompassing usability the software and version control; "clarity", the ability to visualize and communicate to the OPERA team and community the results of this verification exercise. Somewhere between these considerations is also the ability to inspect, experiment with, and modify this verification workflow to understand DSWx better.

There are three components of the repository:

1. The collecton of python routines that are used by the workflow, i.e. the simple statistical functions to sample/compute metrics/etc., the routines that link the various publicly available data including the OPERA provisional products, etc.
2. The notebooks that apply these routines in order to compare OPERA provisoinal data and validation datasets available.
3. The notebooks used to format the results of these comparisons into a human readable format including a pdf slides via Beamer/Latex.

This repository is the descendent of:

+ https://github.com/OPERA-Cal-Val/DSWx-HLS-Requirement-Verification
+ https://github.com/OPERA-Cal-Val/DSWx-HLS-Verification-Latex-Presentation


# Installation

1. Clone this repository
2. `mamba env update -f environment.yml`
3. `conda activate dswx_val`
4. `python -m ipykernell install --user --name --dswx_val`
5. `pip install .` or for development `pip install -e .`
6. `python -m ipykernel install --user --name dswx_val`

Explore the notebooks using jupyter lab. It's *imperative* to run jupyter through the correct environment so that jupytext will be able to load the percent files as notebooks.

## Latex

Need a working version `latex` (specifically, the command line utility `latexmk`). This should be done without `conda`. Here are some instructions by platform: https://mg.readthedocs.io/latexmk.html. For Mac users, as the instructions suggest, we recommend installing [MacTex](https://tug.org/mactex/).

# Setup for Validation Table Generation

The `dswx_verification` package in this repository comes with a geojson table that links:

1. Classified planet imagery
2. DSWx products produced on the validation clone
3. HLS IDs

It also includes relevant urls for these datasets. The geometric extent of each row is determined by the classified planet imagery.

First, one needs to have each of the following items completed to update a table (i.e. when a new validation clone is deployed) to get the latest datasets.

1. JPL VPN access and to be connected to the VPN
2. Have group access to the validation clone (that requires coordination with HySDS to be added to the appropriate LDAP group)
3. Create a `.env` file with JPL credentials.

Specifically, for 3. the `.env` should look like

```
ES_USERNAME='<JPL USERNAME>'
ES_PASSWORD='<JPL PASSWORD>'
```

# Usage

We use `jupytext` to better version control the highly interactive notebooks. There are two routes for exploration. In both cases, we use a `yml` file to share parameters across the different notebooks. A sample can be found [here](notebooks/verification_parameters.yml).

## Individual Notebooks over a given Validation Site

Each site can be explored through [notebooks/1_Verification_Metrics_at_Validation_Site.py](notebooks/1_Verification_Metrics_at_Validation_Site.py). The `notebooks/*.py` files should be viewable in a jupyterlab. Make sure to use the correct environment set up above. When saving the notebook, the file will update `*.py` file automatically.

## Running the Verification Workflow for all Validation Datasets

As above, we need the correct environment (i.e. `dwx_val` dictated by the `environment.yml`) as well as a `yml` file to share parameters across the different pieces of the workflow. An example can be found here: [here](notebooks/verification_parameters.yml).

### A. Generating the Verification data for all Datasets

Navigate to the `notebooks` directory in this repository.

1. `jupytext --set-formats ipynb,py:percent *.py` (generates the corresponding notebook from the percent formatted file)
2. Run `python ../verify_dswx.py --yaml_file verification_parameters.yml`

Adjust the parameters within the yaml file as required. This will create `*.tex` files that can be compiled into a slide deck.

### B. Generating Beamer slides

Navigate to the presentations directory as indicated in the `yaml` file.  Compile a latex document with `latexmk main.tex --pdf`
