# ICEC-vibrational-motion
Interatomic Coulombic Electron Capture (ICEC) is an environment-mediated process in which a free electron attaches to a species by transferring excess energy to a neighbor. 
We incorporated the vibrational motion into an analytical model of the ICEC cross section, including both energy and electron transfer, and applied our theory to (HeNe)+.

- HeNe_input.py : defines the general input parameters for the HeNe+ dimer.
- HeNe-*.py : starts the calculations for the ICEC cross sections for a specific electronic transition. Uses classes and functions defined in icec/.
- plots.py : generates the figures based on the calculated ICEC cross sections.
- icec : contains the classes ICEC, OverlapICEC, InterICEC, OverlapInterICEC which are the base modules for the calculations. Overlap means that the cross section corresponding to the electron transfer mechanism is calculated. icec further contains the file calculations.py which uses the base modules to generate the output files into a folder ./results.
- momenta : contains the values of the wavevectors corresponding to the discretized dissociative Morse states, box of length L = 10 angstrom.
