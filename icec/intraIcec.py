import numpy as np
import matplotlib.pyplot as plt
import copy
from crosssection.icec.constants import *

class IntraICEC:
    """ 
    - degeneracyFactor : g_{A^-} / g_A
    - IP: Ionization potential (eV)
    - PI_xs: Function, Fit for Photoionization cross section (eV -> Mb)
    - prefactor: terms that are neither energy nor R dependent 
    """
    def __init__(self, degeneracyFactor , IP_A, IP_B, PI_xs_A, PI_xs_B) :
        self.degeneracyFactor = degeneracyFactor 
        self.IP_A = IP_A * EV2HARTREE # assumption: adiabatic ionization energy
        self.IP_B = IP_B * EV2HARTREE

        self.PI_xs_A = PI_xs_A # 3 dim vector: [vi,vf,E,xs]
        self.PI_xs_B = PI_xs_B
    
        self.prefactor = (3 * c**2) / (8 * np.pi)

    def make_energy_grid(self, minEnergy=None, maxEnergy=10, resolution=100, geometric=True): 
        """ Make a suitable grid of incoming electron energies.
        - Energy (eV)
        - resolution : number of grid points
        """
        maxEnergy = maxEnergy * EV2HARTREE
        if minEnergy is None:
            minEnergy = self.thresholdEnergy
        else:
            minEnergy = minEnergy * EV2HARTREE        
        if geometric:
            self.energyGrid = np.geomspace(minEnergy, maxEnergy, resolution)
        else:
            self.energyGrid = np.linspace(minEnergy, maxEnergy, resolution)

    def make_R_grid(self, Rmin=2, Rmax=10, resolution=100): 
        """ Make a suitable grid of interatomic distances.
        - R (Bohr, a.u.)
        - resolution : number of grid points
        """
        self.rGrid = np.linspace(Rmin, Rmax, resolution)
        

    def interpolate_PI_xs(self, hbarOmega, PI):
        PI_energy, PI_xs = PI
        return np.interp(hbarOmega, PI_energy, PI_xs)
    
    
    def PI_xs(self, v, vp, hbarOmega):
        # TODO
        return 0
        
    
    def energy_relation(self, electronE, v_A, v_Ap, v_B, v_Bp):
        # TODO
        vib_energy_A = 0 if v_A is None else (Morse_Ap.energy(v_Ap) - Morse_Ap.energy(0)) - (Morse_A.energy(v_A) - Morse_A.energy(0))
        transition_A = self.IP_A + vib_energy_A
        
        vib_energy_B = 0 if v_Bp is None else (Morse_Bp.energy(v_Bp) - Morse_Bp.energy(0)) - (Morse_B.energy(v_B) - Morse_B.energy(0))
        transition_B = self.IP_B - vib_energy_B
        
        hbarOmega = electronE + transition_A
        electronE_f = hbarOmega - transition_B
        
        return hbarOmega, electronE_f
    
    
    # ----- CROSS SECTION -----    
    def xs(self, electronE, R, v_A, v_Ap, v_B, v_Bp):
        """ Calculate cross section (a.u.) of ICEC for some kinetic energy and R.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - R: internuclear distance: (Bohr, a.u.)
        - v_A+ -> v_A (v_A -> v_A+ Photoionization)
        - v_B -> v_B+
        """   
        hbarOmega, electronE_f = self.energy_relation(electronE, v_A, v_Ap, v_B, v_Bp)
        
        if electronE_f <= 0: 
            return 0
        else: 
            # TODO
            PI_xs_A = PI_xs_A(v_A, v_Ap, hbarOmega*HARTREE2EV)*MB2AU
            PI_xs_B = PI_xs_B(v_B,v_Bp, hbarOmega*HARTREE2EV)*MB2AU
            return self.prefactor * self.degeneracyFactor * PI_xs_A * PI_xs_B / (electronE * hbarOmega**2 * R**6)


    def xs_energy(self, R, v_A=None, v_Ap=0, v_B=0, v_Bp=None):
        """ Calculate cross section (Mb) of ICEC for given range of kinetic energies.
        - R: internuclear distance: (Bohr, a.u.)
        """        
        if not hasattr(self, 'energyGrid'):
            self.make_energy_grid()
        xs = np.array([
            self.xs(energy, R, v_A, v_Ap, v_B, v_Bp)
            for energy in self.energyGrid
        ]) 
        return xs * AU2MB

    def xs_R(self, electronE, v_A=None, v_Ap=0, v_B=0, v_Bp=None):
        """ Calculate cross section (Mb) of ICEC for given range of interatomic distances.
        - electronE : energy of incoming electron (eV) 
        - R : interatomic distance (Bohr, a.u.)
        """
        electronE = electronE * EV2HARTREE
        if not hasattr(self, 'rGrid'):
            self.make_R_grid()
        xs = np.array([
            self.xs(electronE, r, v_A, v_Ap, v_B, v_Bp)
            for r in self.rGrid
        ])
        return xs * AU2MB

    def plot_xs(self, ax, xs, label="ICEC", **kwargs):
        '''Plots the Cross section xs [Mb]'''
        ax.plot(self.energyGrid*HARTREE2EV, xs, label=label, **kwargs)
        ax.set_xlabel(r'$E_\text{el}$ [eV]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        ax.set_yscale('log')
        ax.set_title('ICEC cross section')

    def plot_xs_R(self, ax, xs, **kwargs):
        '''Plots the Cross section xs [Mb]'''
        ax.plot(self.rGrid, xs, **kwargs)
        ax.set_xlabel(r'$R$ [a.u.]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        ax.set_yscale('log')
        ax.set_title('ICEC cross section')

    def plot_PR_xs(self, ax, **kwargs):
        '''Plots the Photorecombination Cross section [Mb]'''
        PR_xs = np.array([])
        for electronE in self.energyGrid:
            hbarOmega = electronE + self.IP_A
            PI_xs = self.PI_xs_A(hbarOmega*HARTREE2EV)*MB2AU
            xs = self.degeneracyFactor * hbarOmega**2 / (2*electronE*c**2) * PI_xs
            PR_xs = np.append(PR_xs, [xs * AU2MB])
        ax.plot(self.energyGrid*HARTREE2EV, PR_xs, **kwargs)
        
        