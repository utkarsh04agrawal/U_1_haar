# U_1_haar

This repo is to study monitored circuits with U(1) symmetry. We use Qiskit to run quantum simulations.

entanglement.py and U_1_entanglement.py contains some custom functions to calculate von Neuman entropy, negativity, number entropy etc.

haar_sampler.py contains functions to generate random Haar unitaries. They also take custom random generator as input.


# To Do.
By the end of April have preliminary data

25 April
    [x] Write matchgate sampler
    [x] Write scrambler function
    [x] Run purification with scrambling for haar and matchgate

I realize that Qiskit is too slow as I am saving the statevector at each time step. Going to implement numpy version and sparse version.

27 April
    [x] Numpy version
    [x] Get data

29 April
We find interesting data. For MG circuits proj measurement leads to linear (sublinear?) purification time compared to exp. for weak measurement. To understand this further we need to look at half system entanglement.
    [] Write code to collect half_system entanglement
    [x] Run purification with haar encoder

2 May
    [x] Change MG sampler.
        [] Re-run MG dynamics with proj measurement
        [x] Re-run MG dynamics with weak measurement
        [] Re-run MG dynamics with Haar encoder for proj measurement
        [] Re-run MG dynamics with Haar encoder for weak measurement
    [] Run MG dynamics with varying strength of Haarness
    [] Code to collect half-system entanglement
    [] Run symmetric weak measurement scheme.

7 May
    [x] make gitignore file in data directories
