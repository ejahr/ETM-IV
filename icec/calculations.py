from .icec import ICEC, OverlapICEC
from .interIcec import InterICEC, OverlapInterICEC
from .constants import *
import numpy as np
import re
import os

def get_energies(icec: InterICEC, el_state, max_dissE, box_length):
    '''Retrieve precalculated momenta and convert to energies. 
    Momenta correspond to the continuum states of the morse potential discretized in a box of length box_length.
    - el_state : identifier for the electronic state (e.g. X,A,B for HeNe+)
    - max_dissE : maximum dissociative energy to which the energies of the continuum states were solved.
    - box_length : box length [Angstrom]'''
    file_name = el_state + "_" + str(box_length) + "A-"+ max_dissE + "eV.txt"
    with open("./momenta/"+file_name) as file:
        text = file.read()
        momentas = re.findall(r'k -> ([\d.]+)', text)
        diss_energies = np.array([float(k)**2/(2*icec.Morse_f.mu) for k in momentas])
    return diss_energies # [Hartree, a.u.]

# TODO Precalculate the norm of the dissociative states. Would save computation time
def calculate_norm_diss(icec: InterICEC, el_state, max_dissE, box_length):
    diss_energies = get_energies(icec, el_state, max_dissE, box_length)    
    norms = np.array(list(map(icec.Morse_f.norm_diss, diss_energies)))
    energies_and_norms = np.vstack((diss_energies, norms)) 
    header = "Energies [a.u.] and norms of the continuum Morse states of " + el_state
    file_path = "./continuum/" + el_state + "_" + str(box_length) + "A-"+ max_dissE + "eV.txt" 
    np.savetxt(file_path, np.transpose(energies_and_norms), fmt='%1.3e', header=header)
    
# TODO Retrieve the precalculated norm of the dissociative states
def get_norm_diss(icec: InterICEC, el_state, max_dissE, box_length):
    return None

# ============================================================
# ====================== CROSS SECTION =======================
def xs_const_R(process, header, degeneracy, R, icec:ICEC, overlapicec:OverlapICEC=None):
    new_header = 'ICEC cross section at R = ' + str(R) + ' Bohr, ' + str(round(R*BOHR2ANGSTROM, 5)) + ' Angstrom.\n'
    new_header += header
    if overlapicec is not None:
        new_header += "| E_in [eV] | xs [Mb] | overlap [Mb] |"
    else:
        new_header += "| E_in [eV] | xs [Mb] |"
    
    minEnergy = max(icec.thresholdEnergy*HARTREE2EV, 0.025)
    icec.make_energy_grid(minEnergy)
    xs_array = np.vstack((icec.energyGrid*HARTREE2EV, icec.xs_energy(R)/degeneracy))
    if overlapicec is not None:
        overlapicec.make_energy_grid(minEnergy)
        xs_array = np.vstack((xs_array, overlapicec.xs_energy(R)))

    os.makedirs('./results', exist_ok=True)
    file_path = "./results/" + process + ".xs.constR.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=new_header)

def header_for_xs(header, overlap, vi_range, bc=False):
    new_header = header + "vibrationally resolved ICEC cross section\n"
    new_header += 'Initial bound Morse states: ' + ' '.join(str(vi) for vi in vi_range) + '\n'
    new_header += "| E_in [eV] |"
    if bc: 
        new_header += " b->c [Mb] |"
    else:
        new_header += " b->b [Mb] |" 
    if overlap:
        new_header += " overlap [Mb] |"
    return new_header

def xs_bb(process, header, degeneracy, icec: InterICEC, overlapicec: OverlapInterICEC = None):
    '''Calculates and saves the bound to bound ICEC cross sections.
    - process : string identifying the process (i.e. 1A1.2A1 for X->B for NeHe+) 
    - header : beginning of the header
    - degeneracy : cross section is divided by this number. (E.g. by 3 for B->A for NeHe+)
    TODO unify the degeneracies to be more consistent
    '''
    overlap: bool = overlapicec is not None
    vi_range = [vi for vi in range(icec.Morse_i.vmax+1)]
    new_header = header_for_xs(header, overlap, vi_range)
    xs_array = icec.energyGrid*HARTREE2EV

    for vi in vi_range:
        print("- b-b " + str(vi))
        xs = icec.xs_vi(vi)/degeneracy
        xs_array = np.vstack((xs_array, xs))  # --- -> ===
        if overlap:
            print("- b-b " + str(vi) + ' overlap')
            xs = overlapicec.xs_vi(vi)
            xs_array = np.vstack((xs_array, xs))

    os.makedirs('./results', exist_ok=True)
    file_path = "./results/" + process + ".bb.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=new_header)

def xs_bc(process, header, degeneracy, diss_energies, icec: InterICEC, overlapicec: OverlapInterICEC = None):
    '''Calculates and saves the bound to continuum ICEC cross sections.
    - process : string identifying the process (i.e. 1A1.2A1 for X to B for NeHe+) 
    - header : beginning of the header
    - degeneracy : if the cross section needs to be divided by a number. (E.g. 3 for B->A)
    - diss_energies : energies of the dissociative states [Hartree, a.u.]
    '''
    overlap: bool = overlapicec is not None
    vi_range = [vi for vi in range(icec.Morse_i.vmax+1)]
    new_header = header_for_xs(header, overlap, vi_range, bc=True)
    xs_array = icec.energyGrid*HARTREE2EV

    for vi in vi_range:
        print("- b-c vi=" + str(vi))
        xs = icec.xs_vi_to_continuum(vi, diss_energies)/degeneracy
        xs_array = np.vstack((xs_array, xs))

        if overlap: 
            print("- b-c vi=" + str(vi) + " overlap")
            xs = overlapicec.xs_vi_to_continuum(vi, diss_energies)
            xs_array = np.vstack((xs_array, xs))

    box_length = round(icec.Morse_f.box_length*BOHR2ANGSTROM)
    os.makedirs('./results', exist_ok=True)
    file_path = "./results/" + process + ".bc." + str(box_length) + "A.txt"
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=new_header)
    
# ============================================================
# ===== TEMPERATURE DEPENDENT XS BASED ON CALCULATED XS ======
def read_results_file(process, box_length=10, dir:str='./'):
    '''box_length : [Angstrom]'''
    file_path = dir + "results/" + process + ".bb.txt"
    results_bb = np.loadtxt(file_path, comments='#')
    file_path = dir + "results/" + process + ".bc." + str(box_length) + "A.txt"
    results_bc = np.loadtxt(file_path, comments='#')
    return results_bb, results_bc

def xs_boltzmann(process, T, icec:InterICEC, overlapicec:OverlapInterICEC=None, summed=False):
    ''' Assume energies bb = energies bc
    TODO check if files exist, otherwise start calculation
    ''' 
    results_bb, results_bc = read_results_file(process)
    energies = results_bb[:,0] #eV

    avg_bb, avg_bc, avg = 0, 0, 0
    # Hartree / (Hartree/K) / K
    norm = 1./sum(np.exp(-icec.Morse_i.E(vi)/KB/T) for vi in range(icec.Morse_i.vmax)) # ignore last initial vib state
    
    for vi in range(icec.Morse_i.vmax): # ignore last initial vib state
        if overlapicec is not None:  
            xs_bb = results_bb[:, 2*vi+1] + results_bb[:, 2*vi+2] # Mb
            xs_bc = results_bc[:, 2*vi+1] + results_bc[:, 2*vi+2]
        else:    
            xs_bb = results_bb[:, vi+1] 
            xs_bc = results_bc[:, vi+1]

        avg_bb += np.exp(-icec.Morse_i.E(vi)/KB/T) * xs_bb
        avg_bc += np.exp(-icec.Morse_i.E(vi)/KB/T) * xs_bc
        
        avg += np.exp(-icec.Morse_i.E(vi)/KB/T) * (xs_bb + xs_bc)

    if summed:
        return energies, norm*avg
    else:
        return energies, norm*avg_bb, norm*avg_bc
            
def generate_xs_temperature(process, header, T, icec:InterICEC, overlapicec: OverlapInterICEC = None):
    '''Calculates temperature dependent ICEC cross sections.
    - process : string identifying the process (i.e. 1A1.2A1 for X to B for NeHe+) 
    - header : beginning of the header
    - T : list of temperatures
    '''
    new_header = 'Temperature dependent ICEC cross sections. \nAssumes vi are populated according to the Boltzmann distribution.\n'
    new_header = new_header + header
    new_header +=  "| E_in [eV] | xs [Mb] T = " + str(T) + ' |'
    results_bb, _ = read_results_file(process)
    xs_array = results_bb[:,0]
    for t in T: 
        _, xs = xs_boltzmann(process, t, icec, overlapicec, summed=True)
        xs_array = np.vstack((xs_array, xs))
    
    file_path = "./results/" + process + ".xs.temperature.txt" 
    np.savetxt(file_path, np.transpose(xs_array), fmt='%1.3e', header=new_header) 

# ============================================================
# ========================= SPECTRUM =========================
def spectrum_bc_vi(vi, degeneracy, electronE, diss_energies, icec: InterICEC, overlapicec: OverlapInterICEC = None):
    overlap: bool = overlapicec is not None
    print("- spectrum b-c vi=" + str(vi))
    
    E_out, xs, _ = icec.spectrum_bc(electronE, vi, diss_energies).T
    xs *= 1/degeneracy * MB2AU
    
    electronE_col = np.full((xs.shape[0], 1), electronE)
    vi_col = np.full((xs.shape[0], 1), vi)
    vf_col = np.full((xs.shape[0], 1), -1) # there needs to be some integer

    if overlap:
        print("- spectrum b-c vi=" + str(vi) + " overlap")
        E_out, xs_overlap, _ = overlapicec.spectrum_bc(electronE, vi, diss_energies).T
        xs_overlap *= MB2AU
        return np.column_stack((electronE_col, E_out, xs, xs_overlap, vi_col, vf_col))
    else:
        return np.column_stack((electronE_col, E_out, xs, vi_col, vf_col))
    
def spectrum_bc(degeneracy, electronE, diss_energies, icec: InterICEC, overlapicec: OverlapInterICEC = None):
    spectrum = np.array([]) 
    for vi in range(icec.Morse_i.vmax+1):
        spectrum_vi = spectrum_bc_vi(vi, degeneracy, electronE, diss_energies, icec, overlapicec)
        if spectrum.size == 0:
            spectrum = spectrum_vi
        else:
            spectrum = np.vstack((spectrum, spectrum_vi)) 
    return spectrum 
    
def generate_spectrum_bc(process, header, degeneracy, electronE, diss_energies, icec: InterICEC, overlapicec: OverlapInterICEC = None):
    '''Calculates and saves the spectrum for bound to continuum ICEC.
    - process : string identifying the process (i.e. 1A1.2A1 for X to B for NeHe+) 
    - header : beginning of the header
    - degeneracy : if the cross section needs to be divided by a number. (E.g. 3 for B->A)
    - electronE : incoming electron energy [eV]
    - diss_energies : energies of the dissociative states [Hartree, a.u.]
    '''
    overlap: bool = overlapicec is not None
    new_header = header + "Spectrum for vibrationally resolved bound-continuum ICEC\n"
    if overlap:
        new_header += "| E_in [eV] | E_out [eV] | b->c : overlap [a.u.] | vi | -1 (continuum) |" 
    else:
        new_header += "| E_in [eV] | E_out [eV] | b->c [a.u.] | vi | -1 (continuum) |" 
    spectrum = spectrum_bc(degeneracy, electronE, diss_energies, icec, overlapicec)
    box_length = round(icec.Morse_f.box_length*BOHR2ANGSTROM)
    os.makedirs('./results', exist_ok=True)
    file_path = "./results/" + process + ".bc.spectrum."+ str(electronE) + "eV." + str(box_length) + "A.txt" 
    np.savetxt(file_path, spectrum, fmt='%1.3e', header=new_header)  
    
# ============================================================
def spectrum_bb_vi(vi, degeneracy, electronE, icec: InterICEC, overlapicec: OverlapInterICEC = None):
    overlap: bool = overlapicec is not None
    print("- spectrum b-b vi=" + str(vi))
    
    E_out, xs, vf = icec.spectrum_bb(electronE, vi).T
    xs *= 1/degeneracy * MB2AU

    electronE_col = np.full((xs.shape[0], 1), electronE)
    vi_col = np.full((xs.shape[0], 1), vi)
    
    if overlap:
        print("- spectrum b-b vi=" + str(vi) + " overlap")
        E_out, xs_overlap, vf = overlapicec.spectrum_bb(electronE, vi).T
        xs_overlap *= MB2AU
        return np.column_stack((electronE_col, E_out, xs, xs_overlap, vi_col, vf))
    else:
        return np.column_stack((electronE_col, E_out, xs, vi_col, vf))
    
def spectrum_bb(degeneracy, electronE, icec: InterICEC, overlapicec: OverlapInterICEC = None):
    spectrum = np.array([]) 
    for vi in range(icec.Morse_i.vmax+1):
        spectrum_vi = spectrum_bb_vi(vi, degeneracy, electronE, icec, overlapicec=overlapicec)
        if spectrum.size == 0:
            spectrum = spectrum_vi
        else:
            spectrum = np.vstack((spectrum, spectrum_vi)) 
    return spectrum
            
def generate_spectrum_bb(process, header, degeneracy, electronE, icec: InterICEC, overlapicec: OverlapInterICEC = None):
    '''Calculates and saves the spectrum for bound to bound ICEC.
    - process : string identifying the process (i.e. 1A1.2A1 for X to B for NeHe+) 
    - header : beginning of the header
    - degeneracy : if the cross section needs to be divided by a number. (E.g. 3 for B->A)
    - electronE : incoming electron energy [eV]
    '''
    overlap: bool = overlapicec is not None
    new_header = header + "Spectrum for vibrationally resolved bound-bound ICEC\n"
    if overlap:
        new_header += "| E_in [eV] | E_out [eV] | b->b : overlap [a.u.] | vi | vf |" 
    else:
        new_header += "| E_in [eV] | E_out [eV] | b->b [a.u.] | vi | vf |" 
    spectrum = spectrum_bb(degeneracy, electronE, icec, overlapicec=overlapicec)
    os.makedirs('./results', exist_ok=True)
    file_path = "./results/" + process + ".bb.spectrum."+ str(electronE) + "eV.txt" 
    np.savetxt(file_path, spectrum, fmt='%1.3e', header=new_header) 
    
# ============================================================
def spectrum2d_bc(process, header, degeneracy, energies_in, diss_energies, icec: InterICEC, overlapicec: OverlapInterICEC = None, vi = 0):
    '''Calculates and saves the 2D spectrum for ICEC. NOT SURE IF THIS WORKS CORRECTLY.
    - process : string identifying the process (i.e. 1A1.2A1 for X to B for NeHe+) 
    - header : beginning of the header
    - degeneracy : if the cross section needs to be divided by a number. (E.g. 3 for B->A)
    - energies_in : incoming electron energies [eV]
    - diss_energies : energies of the dissociative states [Hartree, a.u.]
    '''
    overlap: bool = overlapicec is not None
    new_header = header + "Spectrum [Mb] for vibrationally resolved bound-continuum ICEC\n"
    new_header += "First row: energy of incoming electron (E_in), in duplicates \n"
    new_header += "Then for each electron energy: E_out | xs(E_in -> E_out)"

    # ----- asymptotic ------
    print("start 2d spectrum b-c")
    spectrum2d = np.empty((len(diss_energies), 2*len(energies_in)))
    for i in range(len(energies_in)):
        electronE = energies_in[i]
        spectrum = icec.spectrum_bc(electronE, vi, diss_energies)
        spectrum[:,1] *= 1/degeneracy
        spectrum2d[:,2*i:2*i+2] = spectrum[:,:2]
    spectrum2d_bc = np.vstack([np.repeat(energies_in, 2), spectrum2d]) 

    # ----- overlap ------
    if overlap:
        print("start 2d spectrum b-c overlap")
        spectrum2d = np.empty((len(diss_energies), 2*len(energies_in)))
        for i in range(len(energies_in)):
            electronE = energies_in[i]
            spectrum = overlapicec.spectrum_bc(electronE, vi, diss_energies)
            spectrum2d[:,2*i:2*i+2] = spectrum[:,:2]
        spectrum2d_bc_overlap = np.vstack([np.repeat(energies_in, 2), spectrum2d]) 
        
    box_length = round(icec.Morse_f.box_length*BOHR2ANGSTROM)
    
    file_path = "./results/" + process + ".bc.spectrum2d." + str(box_length) + "A.txt" 
    np.savetxt(file_path, spectrum2d_bc, fmt='%1.3e', header=new_header)

    if overlap:
        file_path = "./results/" + process + ".bc.overlap.spectrum2d." + str(box_length) + "A.txt" 
        np.savetxt(file_path, spectrum2d_bc_overlap, fmt='%1.3e', header=new_header)