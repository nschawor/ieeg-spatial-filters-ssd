# Enhancing oscillations in intracranial electrophysiological recordings with data-driven spatial filters

This repository provides analysis code to compute data-driven spatial filters using spatio-spectral decomposition in intracranial electrophysiological data. The repository code recreates results and figures from the following manuscript:

# Reference
Schaworonkow N & Voytek B: [Enhancing oscillations in intracranial electrophysiological recordings with data-driven spatial filters](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009298) _PLoS Computional Biology_ (2021). https://doi.org/10.1371/journal.pcbi.1009298


# Dataset
The results are based on following available openly available data set: [A library of human electrocorticographic data and analyses.](https://exhibits.stanford.edu/data/catalog/zk881ps0522) which is described in detail in following article:

Miller, K.J. A library of human electrocorticographic data and analyses. _Nature Human Behavior_ 3, 1225–1235 (2019). https://doi.org/10.1038/s41562-019-0678-3.

To reproduce the figures from the spatial filters manuscript, the data set should be downloaded and placed in the folder ```data```.

# Requirements

The provided python3 scripts are using ```scipy``` and ```numpy``` for general computation, ```pandas``` for saving intermediate results to csv-files. ```matplotlib``` for visualization. For EEG-related analysis, the ```mne``` package is used. For computation of aperiodic exponents: [```fooof```](https://fooof-tools.github.io/fooof/) and for computation of waveform features: [```bycycle```](https://bycycle-tools.github.io/bycycle/). Specifically used versions can be seen in the ```requirements.txt```.


# Pipeline

To reproduce the figures from the command line, navigate into the ```code``` folder and execute ```make all```. This will run through the preprocessing steps, the computation of spatial filters, the analysis of peak frequencies and the oscillatory burst analysis. The scripts can also be executed separately in the order described in the ```Makefile```.
