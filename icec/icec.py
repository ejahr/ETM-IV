import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from itertools import repeat
from multiprocessing import Pool

# Constants in atomic units
c = 137

# conversion factors for energies
EV2HARTREE = 3.67493e-2
HARTREE2EV = 27.2114
WAVENUMBER2HARTREE = 4.55633e-6

# length
M2BOHR = 18897259885.789 
BOHR2M = 5.2917721941e-11
ANGSTROM2BOHR = 1.8897259886
BOHR2ANGSTROM = 0.529177249

# cross section
MB2M2 = 1e-22
MB2AU = MB2M2 * M2BOHR**2
AU2MB = BOHR2M**2 / MB2M2

## ICEC cross section for atom - atom
# Cross section
# \begin{equation}
#     \sigma (k) = 
#     \frac{3 (\hbar c)^4 }{4 \pi} \,
#     \frac{\sigma_\text{PR}^A(\epsilon_k) \sigma_\text{PI}^B(\omega)}{R^6 (\hbar\omega)^4}
# \end{equation}
# \begin{equation}
#     \sigma (k) = 
#     \frac{3 \hbar^4 c^2}{8 \pi m_e} \,
#     \frac{g_{A^-}}{g_A} \,
#     \frac{\sigma_\text{PI}^{A^-}(\epsilon_k) \sigma_\text{PI}^B(\omega)}
#         {R^6 \epsilon_k(\hbar\omega)^2}
# \end{equation}
# Transferred energy
# \begin{equation}
# \hbar\omega = \epsilon_k + IP_{A}
# \end{equation}

class ICEC:
    """ 
    - EA: Electron affinity of A (ev)
    - IP: Ionization potential of B (eV)
    - PI_xs_A: Function, Fit for Photoionization cross section of A- (eV -> Mb)
    - PI_xs_B: Function, Fit for Photoionization cross section of B (eV -> Mb)

    - thresholdEnergy: threshold energy for ICEC (Hartree, a.u.)
    - prefactor: terms that are neither energy nor R dependent 
    """
    def __init__(self, degeneracyFactor, IP_A, IP_B, PI_xs_A, PI_xs_B) :
        self.degeneracyFactor = degeneracyFactor
        self.IP_A = IP_A * EV2HARTREE
        self.IP_B = IP_B * EV2HARTREE
        self.PI_xs_A = PI_xs_A
        self.PI_xs_B = PI_xs_B
        
        self.thresholdEnergy = max(0, self.IP_B - self.IP_A)
        self.prefactor = (3 * c**2) / (8 * np.pi)

    def make_energy_grid(self, minEnergy=None, maxEnergy=10, resolution=100): 
        """ Make a suitable grid of incoming electron energies.
        - Energy (eV)
        - resolution : number of grid points
        """
        maxEnergy = maxEnergy * EV2HARTREE
        if minEnergy is None:
            minEnergy = self.thresholdEnergy
        else:
            minEnergy = minEnergy * EV2HARTREE
        self.energyGrid = np.arange(minEnergy, maxEnergy, 
                                    (maxEnergy - minEnergy) / resolution, dtype=float)

    def make_R_grid(self, Rmin, Rmax, resolution=100): 
        """ Make a suitable grid of interatomic distances.
        - R (Bohr, a.u.)
        - resolution : number of grid points
        """
        self.rGrid = np.arange(Rmin, Rmax, (Rmax - Rmin) / resolution, dtype=float)

    def calculate_xs(self, electronE, R):
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

    def calculate_xs_energy(self, R):
        """ Calculate cross section (Mb) of ICEC for given range of kinetic energies.
        - R: internuclear distance: (Bohr, a.u.)
        """        
        if not hasattr(self, 'energyGrid'):
            self.make_energy_grid()
        xs = np.array([
            self.calculate_xs(energy, R)
            for energy in self.energyGrid
        ]) 
        return xs * AU2MB

    def calculate_xs_R(self, electronE, Rmin, Rmax):
        """ Calculate cross section (Mb) of ICEC for given range of interatomic distances.
        - electronE : energy of incoming electron (eV) 
        - R : interatomic distance (Bohr, a.u.)
        """
        electronE = electronE * EV2HARTREE
        self.make_R_grid(Rmin, Rmax)
        xs = np.array([
            self.calculate_xs(electronE, R)
            for R in self.rGrid
        ])
        return xs * AU2MB
        
    def define_overlap_parameters(self, a_A, a_B, C, d):
        self.a_A = a_A
        self.a_B = a_B
        self.C = C
        self.d = d

    def calculate_overlap_xs(self, electronE, R, lmax):
        """ Calculate cross section (a.u.) of the overlap contribution.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - R: internuclear distance: (Bohr, a.u.)
        - lmax: upper bound for sum over l -> pick large enough for convergence
        """ 
        electronE_f = electronE + self.IP_A - self.IP_B
        if electronE_f <= 0 :
            return 0
        else: 
            a_AB = self.a_A**2 + self.a_B**2
            C = self.C * np.exp(-abs(self.IP_A-self.IP_B)/self.d)
            sum_l = 0
            for l in range(0,lmax+1):
                K_av = electronE*(self.a_A+R)**2 + electronE_f*(self.a_B+R)**2
                J_l = np.exp(-l*(l+1)/K_av)
                sum_l += (2*l+1) * J_l
            sum_l *= C
            return 32*np.pi / electronE**(3/2.) / np.sqrt(electronE_f) / R**2 * (self.a_A*self.a_B/a_AB)**3 * np.exp(-R**2/a_AB) * sum_l

    def calculate_overlap_xs_energy(self, R, lmax):
        """ Calculate cross section (Mb) of the overlap contribution for given range of kinetic energies.
        - R: internuclear distance: (Bohr, a.u.)
        - lmax: upper bound for sum over l -> convergence
        """    
        overlap_xs = np.array([
            self.calculate_overlap_xs(energy, R, lmax)
            for energy in self.energyGrid
        ])
        return overlap_xs * AU2MB
    
    def calculate_overlap_xs_R(self, electronE, lmax, Rmin, Rmax):
        """ Calculate cross section (Mb) of the overlap contribution for given range of interatomic distances.
        - electronE : energy of incoming electron (eV) 
        - R : interatomic distance (Bohr, a.u.)
        """
        electronE = electronE * EV2HARTREE
        self.make_R_grid(Rmin, Rmax)
        overlap_xs = np.array([
            self.calculate_overlap_xs(electronE, R, lmax)
            for R in self.rGrid
        ])
        return overlap_xs * AU2MB

    def plot_xs(self, ax, xs, label="ICEC", **kwargs):
        '''Plot the Cross section xs [Mb]'''
        ax.plot(self.energyGrid*HARTREE2EV, xs, label=label, **kwargs)
        ax.set_xlabel(r'$E_\text{el}$ [eV]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        ax.set_yscale('log')
        ax.set_title('ICEC cross section')

    def plot_xs_R(self, ax, xs, label="ICEC", **kwargs):
        '''Plot the Cross section xs [Mb]'''
        ax.plot(self.rGrid, xs, label=label, **kwargs)
        ax.set_xlabel(r'$R$ [a.u.]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        ax.set_yscale('log')
        ax.set_title('ICEC cross section')

    def plot_PR_xs(self, ax, label="PR", linestyle='dashed', **kwargs):
        '''Plot the Photorecombination Cross section [Mb]'''
        PR_xs = np.array([])
        for electronE in self.energyGrid:
            hbarOmega = electronE + self.IP_A
            PI_xs = self.PI_xs_A(hbarOmega*HARTREE2EV)*MB2AU
            xs = self.degeneracyFactor * hbarOmega**2 / (2*electronE*c**2) * PI_xs
            PR_xs = np.append(PR_xs, [xs * AU2MB])
        ax.plot(self.energyGrid*HARTREE2EV, PR_xs, label=label, linestyle=linestyle, **kwargs)