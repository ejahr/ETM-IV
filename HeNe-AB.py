# He Ne+(2px^-1) -> He+ Ne | A -> B | 1B1 -> 2A1
# The asymptotic cross section is NOT divided by 3, 
# since sigma^PI_px = 1/3*sigma^PI for Neon is cancelled by g_Ne+(2px^-1) = 3*g_Ne+ 

import sys
import numpy as np
from HeNe_input import * # defines  parameters
# declare path to folder containing module crosssection
sys.path.insert(0, '/mnt/home/elena/icec_project')
sys.path.insert(0, '/home/elena/icec-project')
from crosssection.icec.icec import ICEC
from crosssection.icec.interIcec import InterICEC
from crosssection.calculations import get_energies, xs_const_R, generate_xs_temperature, generate_spectrum_bb, generate_spectrum_bc, spectrum2d_bc

header = "He Ne+(2px^-1) -> He+ Ne  | A -> B | 1B1 -> 2A1\n"

process = "1B1.2A1"
el_state = "B"
max_dissE = "8"
max_dissE_for_calc = 2 * EV2HARTREE
box_length = 10 #angstrom

min_kinE = 3        #eV
max_kinE = 10       #eV
num = 100           #number of grid points

electron_energies = [5, 9] #eV for spectrum

degeneracy = 1.
# no overlap

# Change to True for corresponding calculations 
crosssection = False
temperature = False
spectrum = False
spectrum2d = False

icecR = ICEC(*inputNeHe)
icec = InterICEC(*inputNeHe)
icec.define_Morse_i(*stateA)
icec.define_Morse_f(*stateB)  

icec.make_energy_grid(min_kinE, max_kinE, num)
icec.Morse_f.define_box(box_length*ANGSTROM2BOHR)
diss_energies = get_energies(icec, el_state, max_dissE, box_length)
diss_energies = diss_energies[diss_energies < max_dissE_for_calc]

# no overlap
# no overlap
# no overlap 
# no overlap

header_bc = header + "box length = " + str(box_length) + "A, "
header_bc += "maximum dissociative energy = " + str(round(max_dissE_for_calc*HARTREE2EV)) + " eV\n"    
  
# ============================================================
# ====================== CROSS SECTION =======================
if crosssection:
    xs_const_R(process, header, degeneracy, icec.Morse_i.re, icecR)
    #xs_bb(process, header, degeneracy, icec)
    #xs_bc(process, header_bc, degeneracy, diss_energies, icec)
    
if temperature:
    T = [15,77,298]
    generate_xs_temperature(process, header_bc, T, icec)

# ============================================================
# ========================= SPECTRUM =========================
if spectrum:
    for electronE in electron_energies:
        generate_spectrum_bb(process, header, degeneracy, electronE, icec)
        generate_spectrum_bc(process, header_bc, degeneracy, electronE, diss_energies, icec)

if spectrum2d:
    energies_in = np.linspace(min_kinE, max_kinE, num)
    spectrum2d_bc(process, header, degeneracy, energies_in, diss_energies, icec)
