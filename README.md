# Analytical energy and electron transfer model for ICEC with vibrational motion
Interatomic Coulombic Electron Capture (ICEC) is an environment-mediated process in which a free electron attaches to a species by transferring excess energy to a neighbor. 
This implentation incorporates the vibrational motion into an analytical model of the ICEC cross section, including both energy and electron transfer.

[![DOI](https://zenodo.org/badge/977628832.svg)](https://doi.org/10.5281/zenodo.15348382)

## Files
- `HeNe_input.py` : defines the general input parameters for the HeNe+ dimer.
- `HeNe-*.py` : starts the calculations for the ICEC cross sections for a specific electronic transition. Uses classes and functions defined in `icec/`. Results in [_10.6084/m9.figshare.28892498_](https://doi.org/10.6084/m9.figshare.28892498). Run with `python HeNe-AB.py` as an example.
- `plots.py` : generates figures based on the calculated ICEC cross sections. Results in [_10.6084/m9.figshare.28883378_](https://doi.org/10.6084/m9.figshare.28883378). Needs r-matrix results in directory `./r-matrix` for some figures, e.g. [_10.6084/m9.figshare.28883423_](https://doi.org/10.6084/m9.figshare.28883423).
- `icec` : contains the classes ICEC, OverlapICEC, InterICEC, OverlapInterICEC which are the base modules for the calculations. Overlap means that the cross section corresponding to the electron transfer mechanism is calculated. icec further contains the file calculations.py which uses the base modules to generate the output files into a folder `./results`.
- `momenta` : contains the values of the wavevectors corresponding to the discretized dissociative Morse states, box of length L = 10 angstrom.

## Runtime
- `HeNe-AB.py` and `HeNe-BA.py` contain only the energy transfer contribution. The calculations are fairly quick (sub 1h).
- `HeNe-XB.py` and `HeNe-BX.py` can run for more than a day, even though its parallelized. The electron transfer (OverlapInterICEC) is calculation heavy due to l numerical integrations per data point.

## Requirements
- Can be run on Linux or WSL. Not tried on native Windows or iOS.
- Python >= 3.11.10
- Libraries: `numpy`, `scipy`, [`mpmath`](https://mpmath.org/), `matplotlib`, `itertools`, `multiprocessing`
