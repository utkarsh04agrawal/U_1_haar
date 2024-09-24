# U_1_haar

This repo is to study monitored circuits with U(1) symmetry.

entanglement.py and U_1_entanglement.py contains custom functions to calculate von Neuman entropy, negativity, number entropy etc.

haar_sampler.py contains functions to generate random Haar unitaries. They also take custom random generator as input.

There are in total 4 options for the circuit depending on whether there is particle-hole (PH) symmetry in unitary (U) and measurements (M).
- The parent directory contains code for  no PH in U or M

- particle_hole_symmetry contains data  for when U is PH but not M

- symmetric_measurements contain data for with PH in M



# To Do.
By the end of April have preliminary data

25 April
- [x] Write matchgate sampler
- [x] Write scrambler function
- [x] Run purification with scrambling for haar and matchgate

I realize that Qiskit is too slow as I am saving the statevector at each time step. Going to implement numpy version and sparse version.

27 April
- [x] Numpy version
- [x] Get data

29 April
We find interesting data. For MG circuits proj measurement leads to linear (sublinear?) purification time compared to exp. for weak measurement. To understand this further we need to look at half system entanglement.
- [x] Write code to collect half_system entanglement
- [x] Run purification with haar encoder

2 May
- [x] Change MG sampler.
    - [x] Re-run MG dynamics with proj measurement
    - [x] Re-run MG dynamics with weak measurement
    - [ ] Re-run MG dynamics with Haar encoder for proj measurement
    - [ ] Re-run MG dynamics with Haar encoder for weak measurement
- [x] Code to collect half-system entanglement

7 May
- [x] make gitignore file in data directories

8 May
- [ ] Analyze data for half system for non PH symmetric U and M
- [ ] Analyze data for proj measurement purification
- [ ] Run symmetric weak measurement scheme.
- [ ] Run MG dynamics with varying strength of Haarness


17 May
    Created separate package "Circuits" for circuit evolution functions
- [x] Update evolution_utils to import this new package

18 May
Changed the structure of the project significantly. Major changes:
- Made the key functions common. Now there is single evolution_utils,collect_utils,analysis_utils file in the root directory. These contains functions for helping in evolution, collection and saving of data, and analysis of the data, respectively.
- The sub-directories now correspond to different physical scenarios, like, with PH or no PH; with generalized POVM etc.
To do:
- [ ] Collect data for PH (weak and proj)
- [ ] Use generalized_measurement function of the Circuits package to implement a symmetric version of the weak measurements.
