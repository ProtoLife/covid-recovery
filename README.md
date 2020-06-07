

<p>
<a href="https://daptics.ai">
<img src="images/dapticslogotag.png" width="308" height="104" />
</a>
</p>

# Covid-recovery

The aim of this repository is to build models to fit policies for covid-19 response to the results.  If the models work, they may be used to predict efficacious policy modifications.

The project is described in [project.ipynb](https://github.com/ProtoLife/covid-recovery/blob/master/project.ipynb).  NB: the jupyter notebook is merely used to provide documentation that can include equations, since github markdown cannot do so conveniently.

# Origins

The idea of using [daptics](https://daptics.ai) tools to model covid-19 response protocols came from conversations with [John McCaskill](http://biomip.org).  The general idea is out in the world (e.g. Peter's [find on Twitter](https://twitter.com/btshapir/status/1258385835562536964?s=21)).  Daptics tools seem particularly appropriate given their focus on *small data* model building, with all the necessary careful regularization (see the [daptics white paper](https://daptics.ai/pdf/White.pdf) on this).  

# Data

This repo has data sourced externally.  Data curation is not the primary aim of this site, though data quality control (via multiple sources) may eventually be necessary.

Currently this repo has data from

* [Our World in Data](https://github.com/owid/covid-19-data)  (OWID), included as a submodule, covid-19-owid.

* [Oxford Covid-19 Government Response Tracker](https://github.com/OxCGRT/covid-policy-tracker) (OxCGRT) included as a submodule, covid-policy-tracker.

* [Johns Hopkins data](https://github.com/CSSEGISandData/COVID-19.git), included as submodule covid-19-JH

* [UN population data](https://population.un.org) in folder `data`

The first three are updated by executing a script:
```
sh updateSubmodules.sh
```
following initialization (caution: over 1 GB of data) if required using the script:
```
sh initSubmodules.sh
```
The UN population data is static over the time period of the first two, so it should not need updating; for reference the script to download the data is in `data/getUNpopulation.sh`

As other data is added to the repo, the additions should be documented here.

# Notebooks

Check in the Notebooks folder for notebooks. Note that there are some notebooks for covid-response modeling, others for epidemiological modeling.

# Other resources

This project would benefit from any policy databases analogous to the [Oxford Covid-19 Government Response Tracker](https://github.com/OxCGRT/covid-policy-tracker), but with finer-grained data than by country.  State-level information does exist, but so far (June 6, 2020) not reduced to a numerical vector as for OxCGRT.

- [Multistate.us](https://www.multistate.us/pages/covid-19-policy-tracker)
- [Ancor.org](https://www.ancor.org/covid-19/state-tracker)
- [Kaiser](https://www.kff.org/health-costs/issue-brief/state-data-and-policy-actions-to-address-coronavirus/)
- [National Conference of State Legislatures](https://www.ncsl.org/research/health/state-action-on-coronavirus-covid-19.aspx)

Looking for volunteers to transform state-level data (for US, but also for other countries) into the Oxford format!


