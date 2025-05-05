import numpy as np
import copy
from .constants import *

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
    """ICEC cross section
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

    def make_energy_grid(self, minEnergy=None, maxEnergy=10, num=100, geometric=True): 
        """ Make a suitable grid of incoming electron energies.
        - Energy (eV)
        - num : number of grid points
        # TODO add function where you can define the energy grid directly
        """
        maxEnergy = maxEnergy * EV2HARTREE
        if minEnergy is None:
            minEnergy = self.thresholdEnergy
        else:
            minEnergy = minEnergy * EV2HARTREE        
        if geometric:
            self.energyGrid = np.geomspace(minEnergy, maxEnergy, num)
        else:
            self.energyGrid = np.linspace(minEnergy, maxEnergy, num)

    def make_R_grid(self, Rmin=2, Rmax=10, num=100): 
        """ Make a suitable grid of interatomic distances.
        - R (Bohr, a.u.)
        - num : number of grid points
        """
        self.rGrid = np.linspace(Rmin, Rmax, num)
        
    def energy_relation(self, electronE):
        hbarOmega = electronE + self.IP_A
        electronEf = hbarOmega - self.IP_B 
        return hbarOmega, electronEf

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

    def xs_energy(self, R):
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

    def xs_R(self, electronE):
        """ Calculate cross section (Mb) of ICEC for given range of interatomic distances.
        - electronE : energy of incoming electron (eV) 
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
        mask = PR_xs>0
        ax.plot(self.energyGrid[mask]*HARTREE2EV, PR_xs[mask], **kwargs)
        
        
class OverlapICEC(ICEC):
    @classmethod
    def from_ICEC(cls, InstanceICEC: ICEC):
        ''' generate an instance of OverlapICEC from an instance of ICEC'''
        new_inst = copy.deepcopy(InstanceICEC) 
        new_inst.__class__ = cls
        return new_inst
    
    def define_overlap_parameters(self, a_A, a_B, C, d, lmax=10, gaussian_type='s'):
        ''' Defines the overlap parameters, including fitting parameters
        - lmax: upper bound for sum over l -> set large enough for convergence
        '''
        self.a_A = a_A
        self.a_B = a_B
        self.C = C
        self.d = d
        self.lmax = lmax
        self.gaussian_type = gaussian_type

    def Sab(self, R):
        """ Square of the overlap integral of two Gaussians
        """
        a_AB = self.a_A**2 + self.a_B**2
        if self.gaussian_type == 'pz':
            factor = 16 * self.a_A**3 * (self.a_B / a_AB)**5 * R**2
        elif self.gaussian_type == 's':
            factor = (2 * self.a_A * self.a_B / a_AB)**3
        else:
            print('Invalid gaussian type')
            return 0
        return factor* np.exp(-R**2/a_AB)

    def xs(self, electronE, R):
        """ Calculates cross section (a.u.) of the overlap contribution.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - R: internuclear distance: (Bohr, a.u.)
        """ 
        electronE_f = electronE + self.IP_A - self.IP_B
        if electronE_f <= 0 :
            return 0
        else: 
            # overlap of the continuum electrons
            C = self.C * np.exp(-abs(self.IP_A-self.IP_B)/self.d)
            sum_l = 0
            for l in range(0,self.lmax+1):
                K_av = electronE*(self.a_A+R)**2 + electronE_f*(self.a_B+R)**2
                J_l = np.exp(-l*(l+1)/K_av)
                sum_l += (2*l+1) * J_l
            sum_l *= C
            # cross section
            return 4*np.pi / electronE**(3/2) / np.sqrt(electronE_f) / R**2 * self.Sab(R) * sum_l