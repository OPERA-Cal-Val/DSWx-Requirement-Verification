# DSWx-Requirement-Verification

This respository contains the workflow used to verify the DSWx suite. It contains the functions and workflow to verify the OPERA project requirements. The workflow balances reproducibility and clarity. Here, "reproducibility" encompassing usability the software and version control; "clarity", the ability to visualize and communicate to the OPERA team and community the results of this verification exercise. Somewhere in between these two considerations is also the ability to inspect, experiment with, and modify this verification workflow to understand DSWx better.


# Installation

1. Clone this repository
2. `mamba env update -f environment.yml`
3. `conda activate dswx_val`
4. `python -m ipykernell install --user --name --dswx_val`
5. `pip install .` or for development `pip install -e .`
6. `python -m ipykernel install --user --name dswx_val`

Explore the notebooks using jupyter lab.

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

We use `jupytext` to better version control the highly interactive notebooks.

## Individual Notebooks

The `notebooks/*.py` files should be viewable in a jupyter environment. Saving the file will generate a `*.ipynb` file automatically.

## Running the Verification Workflow for all Validation Datasets

### A. Generating the Verification data for all Datasets

1. `jupytext --set-formats ipynb,py:percent notebooks/*.py` (generates the corresponding notebook from the percent formatted file)
2. Run `python verify_dswx.py`

### B. Generating Beamer slides

1. Run `make`
