# Code to reproduce decoding analysis in Gauld and Bao et al: A frontal motor circuit for economic decisions and actions

This repository contains code to reproduces analysis and plots in the paper relating to decoding analysis:

Gauld and Bao et al: A frontal motor circuit for economic decisions and actions

It produces the following plots in the paper

 - Figure 2i 
 - Figure 2k 
 - Figure S5e
 - Figure S5f 
 - Figure S5g 
 - Figure S5h 
 - Figure S5i 


# Decoding analysis instructions 

## Specifying location of code and data

Go to `config.py` and edit the following variables:

- `PROJ_ROOT` : Path to the folder containing this code, eg. /home/yourname/riskychoice-paper-code
- `INTERIM_DATA_DIR` : Path to folder containing widefield (.mat) and behaviour (.csv) files (see Data section below)
- `ALLEN_DATA_DIR` : Path to folder containing the allen dorsal mat file: `allenDorsalMap.mat`
- `PROCESSED_DATA_DIR` : Path to the folder that you want to save processed data, or optionally, if you have already downloaded the processed data, the path to that folder
- `FIGURES_DIR` : Path to the folder to save figures

## Data 

This code assumes the user already have pre-processed files in `INTERIM_DATA_DIR`, they should have names such as:

`OLI-M-0041_20231023_pixel_ave.mat` and `OLI-M-0041_20231023_behaviour.csv`

If not, then the code will generate them from another set of pre-processed files, which should have names such as: 

`OLI-M-0022_20231107_all_data_nodecon_updated_hemocorr_filter.mat`

and from the allen atlas file: `allenDorsalMapSM.mat`, and the ROI mask file: `oli_generic_allen_roimask.mat`

This step requires having MATLAB and npy-matlab installed. Their path should 
be specified in `config.py`

You can obtain npy-matlab from: https://github.com/kwikteam/npy-matlab

## Generating plots for the figures 

Assuming you have the analyzed data (in `PROCESSED_DATA_DIR` or `PROCESSED_DATA_DIR/decoding`) in the form:

`IAA-1125880_all_None_locaNMF_decoding_AirPuffLeft_using_pixel_ave_20251028-riskychoice.npz`

and 

`IAA-1125880_all_None_locaNMF_decoding_AirPuffLeft_using_pixel_ave_20251028-riskychoice-single-region.npz`

You can run the code to generate the plots directly by following the instructions below

### Linux / macOS

```
chmod +x make_plots.sh
```

```
./make_plots.sh
```

### Windows 


Open PowerShell, then do: 

```
.\make_plots.ps1
```

## Running the analysis

### Linux / macOS 

First make the bash script executable:

```
chmod +x run_all.sh
```

Then do 

```
./run_all.sh
```

### Windows 

Open PowerShell, then do: 

```
.\run_all.ps1
```

### Running individual analysis 

The provided bash script runs all the analysis and plots sequentially. If you only want to run part of the analysis, then:

to run only the multiregion decoding analysis

`uv run python train.py --process-name window-decoding`

to run only the single-region decoding analysis 
`uv run python train.py --process-name single-region-decoding`

and if you already have the processed data in the processed data folder (specified in `config.py` as `PROCESSED_DATA_DIR`), 
then run the following to make the plots:

```
uv run python plots.py --process-name plot-windowed-decoding
uv run python plots.py --process-name plot-single-region-decoding
```


### Code run times 

- On a computer with AMD Ryzen 7 8745 HS and 28 GB Ram, the multiregion decoding analysis took about 30 minutes
- The single region decoding analysis took about 2 hrs
- Plotting should be done within two minutes

### Processed data details 

- By default, processed data will be saved in `{PROJ_ROOT}/processed_data/decoding`
- Decoding using activity from all areas together will be named in the form: `OLI-M-0057_all_None_locaNMF_decoding_LeftChoice_using_pixel_ave_20251028-riskychoice.npz`
- Decoding using activity from single areas will be named in the form:  `OLI-M-0021_all_None_locaNMF_decoding_LeftChoice_using_pixel_ave_20251028-riskychoice-single-region.npz`


# TODO

 - Figure out where to put processed data / decoding results data
   - eg. OLI-M-0041_20231023_pixel_ave.mat
 - Work on plotting from processed data?
 - Test on Windows computer
