import numpy as np
import matplotlib.pyplot as plt
import copy
from crosssection.icec.constants import *

class IntraICEC:
    """ 
    - EA: Electron affinity of A (ev)
    - IP: Ionization potential of B (eV)
    - PI_xs_A: Function, Fit for Photoionization cross section of A- (eV -> Mb)
    - PI_xs_B: Function, Fit for Photoionization cross section of B (eV -> Mb)

    - thresholdEnergy: threshold energy for ICEC (Hartree, a.u.)
    - prefactor: terms that are neither energy nor R dependent 
    """
    def __init__(self, degeneracy_i, IP_A, IP_B, PI_xs_A, PI_xs_B) :
        self.degeneracy_i = degeneracy_i
        self.IP_A = IP_A * EV2HARTREE
        self.IP_B = IP_B * EV2HARTREE
        self.PI_xs_A = PI_xs_A
        self.PI_xs_B = PI_xs_B
        
        self.thresholdEnergy = max(0, self.IP_B - self.IP_A)
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
        
    def threshold_energy():
        E_threshold_ICEC = np.array([])

        if A_dimer == True and B_dimer == False:
            for EA in A_EA:
                new_element = B_IP - EA
                E_threshold_ICEC = np.append(E_threshold_ICEC, new_element)
            E_threshold_ICEC[E_threshold_ICEC < 0] = 0
        else:
            E_threshold_ICEC = B_IP - A_EA
        if B_dimer == True:
            E_threshold_ICEC[E_threshold_ICEC < 0] = 0
        else:
            if E_threshold_ICEC < 0:
                E_threshold_ICEC = 0

    def interpolate_PI_xs(self, electronE, PI):
        PI_energy, PI_xs = PI
        return np.interp(electronE + self.A_EA, PI_energy, PI_xs)

    # ----- CROSS SECTION -----    
    def xs(self, electronE, R):
        """ Calculate cross section (a.u.) of ICEC for some kinetic energy and R.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - R: internuclear distance: (Bohr, a.u.)
        """   
        if electronE < self.thresholdEnergy: 
            return 0
        else: 
            hbarOmega = electronE + self.IP_A
            PI_xs_A = self.PI_xs_A(hbarOmega*HARTREE2EV)*MB2AU
            PI_xs_B = self.PI_xs_B(hbarOmega*HARTREE2EV)*MB2AU
            return self.prefactor * self.degeneracyFactor * PI_xs_A * PI_xs_B / (electronE * hbarOmega**2 * R**6)
        

    def xs(self, electronE, R, vf_A, vf_B):
        omega = electronE + IP_A
        PI_xs_A = self.PI_xs_A(vf, vi, E)
        PI_xs_B =  self.PI_xs_B(vi, vf)
        return self.prefactor * self.degeneracyFactor * PI_xs_A * PI_xs_B / (electronE * omega**2 * R ** 6)

    def xs_energy(self, R, vf_A, vf_B):
        """ Calculate cross section (Mb) of ICEC for given range of kinetic energies.
        - R: internuclear distance: (Bohr, a.u.)
        """        
        if not hasattr(self, 'energyGrid'):
            self.make_energy_grid()
        xs = np.array([
            self.xs(energy, R)
            for energy in self.energyGrid
        ]) 
        return xs * AU2MB

    def xs_R(self, electronE, vf_A, vf_B):
        """ Calculate cross section (Mb) of ICEC for given range of interatomic distances.
        - electronE : energy of incoming electron (eV) 
        - R : interatomic distance (Bohr, a.u.)
        """
        electronE = electronE * EV2HARTREE
        if not hasattr(self, 'rGrid'):
            self.make_R_grid()
        xs = np.array([
            self.xs(electronE, R)
            for R in self.rGrid
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
        
        