# He+ Ne -> He Ne+(2pz^-1) | B -> X | 2A1 -> 1A1
# The asymptotic cross section is divided by 3
# to account for sigma^PI_px = 1/3*sigma^PI for Neon

import numpy as np
from HeNe_input import * # defines constants and parameters
from icec.icec import ICEC, OverlapICEC
from icec.interIcec import InterICEC, OverlapInterICEC
from icec.calculations import get_energies, xs_const_R, xs_bb, xs_bc, generate_xs_temperature, generate_spectrum_bb, generate_spectrum_bc, spectrum2d_bc

header = "He+ Ne -> He Ne+(2pz^-1) | B -> X | 2A1 -> 1A1\n"

process = "2A1.1A1"
el_state = "X"
max_dissE = "8"
max_dissE_for_calc = 1 * EV2HARTREE
box_length = 10 #angstrom

min_kinE = 0.05     #eV
max_kinE = 5        #eV
num = 50            #number of grid points

electron_energies = [1] #eV for spectrum

degeneracy = 3.
lmax = 10

# Change to 1 to enable corresponding calculations 
crosssection = 1
temperature = 1
spectrum = 1
spectrum2d = 0

icecR = ICEC(*inputHeNe)
icec = InterICEC(*inputHeNe)
icec.define_Morse_i(*stateB)
icec.define_Morse_f(*stateX)  

icec.make_energy_grid(min_kinE, max_kinE, num)
icec.Morse_f.define_box(box_length*ANGSTROM2BOHR)
diss_energies = get_energies(icec, el_state, max_dissE, box_length)
diss_energies = diss_energies[diss_energies < max_dissE_for_calc]

overlapicecR : OverlapICEC = OverlapICEC.from_ICEC(icecR)
overlapicecR.define_overlap_parameters(*overlapHeNe, lmax)
overlapicec : OverlapInterICEC = OverlapInterICEC.from_InterICEC(icec)
overlapicec.define_overlap_parameters(*overlapHeNe, lmax)

header_bc = header + "box length = " + str(box_length) + "A, "
header_bc += "maximum dissociative energy = " + str(round(max_dissE_for_calc*HARTREE2EV)) + " eV\n"    
 
# ============================================================
# ====================== CROSS SECTION =======================
if crosssection:
    xs_const_R(process, header, degeneracy, icec.Morse_i.re, icecR, overlapicecR)
    xs_bb(process, header, degeneracy, icec, overlapicec=overlapicec)
    xs_bc(process, header_bc, degeneracy, diss_energies, icec, overlapicec=overlapicec)
    
if temperature:
    T = [15,77,298]
    generate_xs_temperature(process, header_bc, T, icec, overlapicec)

# ============================================================
# ========================= SPECTRUM =========================
if spectrum:
    for electronE in electron_energies:
        generate_spectrum_bb(process, header, degeneracy, electronE, icec, overlapicec)
        generate_spectrum_bc(process, header_bc, degeneracy, electronE, diss_energies, icec, overlapicec)

if spectrum2d:
    energies_in = np.linspace(min_kinE, max_kinE, num)
    spectrum2d_bc(process, header, degeneracy, energies_in, diss_energies, icec, overlapicec=overlapicec)