

<p>
<a href="https://daptics.ai">
<img src="images/dapticslogotag.png" width="308" height="104" />
</a>
</p>

# Covid-recovery

The aim of this repository is to build models to fit policies for covid-19 response to the results.  If the models work, they may be used to predict efficacious policy modifications.

The project is described in [project.ipynb](https://github.com/ProtoLife/covid-recovery/blob/master/project.ipynb).  NB: the jupyter notebook is merely used to provide documentation that can include equations, since github markdown cannot do so conveniently.

# Data

This repo has data sourced externally.  Data curation is not the primary aim of this site, though data quality control (via multiple sources) may eventually be necessary.

Currently this repo has data from

* [Our World in Data](https://github.com/owid/covid-19-data)  (OWID), included as a submodule, covid-19-data.

* [Oxford Covid-19 Government Response Tracker](https://github.com/OxCGRT/covid-policy-tracker) (OxCGRT) included as a submodule, covid-policy-tracker.

* [UN population data](https://population.un.org) in folder `data`

The first two are updated by executing a script:
```
sh updateSubmodules.sh
```
The UN population data is static over the time period of the first two, so it should not need updating; for reference the script to download the data is in `data/getUNpopulation.sh`

As other data is added to the repo, the additions should be documented here.

# Notebooks

Check in the Notebooks folder for notebooks.  Currently there is one, that just begins to look at the data.

