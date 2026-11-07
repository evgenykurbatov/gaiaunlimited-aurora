# The Realm of Aurora


This archive contains the entire codebase required to reproduce the results presented in the Paper

* Evgeny P. Kurbatov, Vasily Belokurov, Sergey E. Koposov, Andrey Kravtsov, Elliot Y. Davies, Anthony G. A. Brown, Tristan Cantat-Gaudin, Alfred Castro-Ginard, Andrew R. Casey, Ronald Drimmel, Morgan Fouesneau, Shourya Khanna, Hans-Walter Rix, Alex Wallace. _The realm of Aurora. Density distribution of metal-poor giants in the heart of the Galaxy_. Accepted in MNRAS (2026)

The Paper is a part of the [GaiaUnlimited](https://gaia-unlimited.org/) project dedicated to determining the [Gaia](https://www.esa.int/Science_Exploration/Space_Science/Gaia_overview) Survey Selection Function.


## Structure of the project

The root directory of the codebase is `Aurora/`. The Python-based project pipeline is to be executed in two stages: preparing datasets and the second stage involves applying models to the datasets. The scripts to be run have names beginning with a number and must be run sequentially. Scripts whose names begin with the same number may be run in any order. The first-stage scripts are located in the root directory of the codebase.

The pipeline:

+ `00_arc23rgb.py`
  + Evaluate extinctions for every RGB star in the Andrae et al. (2023) catalogue using `dustmaps` package and the Fitzpatrick et al. (2019) extinction model
+ `0_arc23aurgb.py`
  + Clean the catalogue of globular clusters using the Vasiliev & Baumgardt (2021) catalogue
  + Make metallicity cuts
  + Transform the stellar positions and velocities to Galactocentric frame
+ `0_rgb_iso.py`
  + Calculate the total mass of the RGB stellar population within the `minimint` isochrones of various ages and metallicity values
+ `0_setup_domain.py`
  + Initialise the spatial and magnitude domains
+ `1_extinctions.py`
  + Evaluate celestial extinction maps
+ `1_query_stats.py`
  + Fetch the statistics on XP stars, parallax errors, and visibility periods
+ `2_plxerr_nvis_g.py`
  + Make a model for parallax error vs number of visits and apparent magnitude
+ `2_selfun_xp.py`
  + Using the `gaiaunlimited` package, create a model for the XP selection function vs HEALPixel and apparent magnitude
+ `3_transfun.py`
  + Calculate the transformation function for probability

The rest of the pipeline is a particular model for the Aurora and GSE populations:

+ `pl-nogse-*`
  + No GSE. The numeric index at the end of the directory name corresponds to the metallicity bin; refer to the `config.py` for more information
+ `pl-sc-*`
  + The 'SC' model is applied to the GSE
+ `pl-scd-*`
  + The 'SC+D' model is applied
+ `pl-freesc-*`
  + The geometric parameters of the GSE are free and optimised alongside the parameters of the Aurora

The scripts within these directories should also be run in sequence.

Adjust global settings with `global_config.py`, and local settings with `pl-*/config.py`

All datasets needed to actually run the code are at https://doi.org/10.5281/zenodo.18757406
