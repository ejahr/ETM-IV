import numpy as np
import scipy as sp
from crosssection.icec.constants import *


class Morse:
    """Initialize the Morse model for a diatomic molecule in atomic units (hbar=1, me=1, hartree energy=1).
    Adapted from https://scipython.com/blog/the-morse-oscillator and https://liu-group.github.io/Morse-potential
    Input arguments
    - mu: reduced mass (electron mass)
    - we: Morse parameter (cm-1)
    - req: Equilibrium bond distance (Angstrom)
    - De: Dissociation energy (cm-1)
    - E0: Equilibrium energy E(req) (cm-1)
    Class parameters
    - alpha: Morse parameter
    - lam: Morse parameter (not to be confused with lambda function)
    - vmax: maximum vibrational quantum number
    - rmin: r where V(r) = De on repulsive edge
    - rmax: r where V(r) = f*De, f<1
    """

    def __init__(self, mu, we, req, De, E0=0):
        self.mu = mu  # in electron mass
        self.we = we 
        self.req = req 
        self.De = De 
        self.E0 = E0 

        self.alpha = self.we * np.sqrt(self.mu / 2 / self.De)
        self.lam = np.sqrt(2 * self.mu * self.De) / self.alpha
        self.z0 = 2 * self.lam * np.exp(self.alpha * self.req)
        self.vmax = int(self.lam - 0.5)

        self.rmin = self.req - np.log(2) / self.alpha
        f = 0.99
        self.rmax = self.req - np.log(1 - f) / self.alpha

    def V(self, r):
        """Morse potential
        - r : interatomic distance (Bohr, a.u.)
        """
        return self.De * (1 - np.exp(-self.alpha * (r - self.req))) ** 2 - self.De

    def psi(self, v, r):
        """v-th eigenstate of the Morse potential
        - r : interatomic distance (Bohr, a.u.)
        """
        z = self.z0 * sp.exp(-self.alpha * r)
        N = sp.sqrt(
            (2 * self.lam - 2 * v - 1)
            * sp.factorial(v)
            * self.alpha
            / sp.gamma(2 * self.lam - v)
        )
        return (
            N
            * z ** (self.lam - v - 0.5)
            * sp.exp(-z / 2)
            * sp.laguerre(v, 2 * self.lam - 2 * v - 1, z)
        )

    def E(self, v):
        """Energy [Hartree] of the v-th (bound) Morse state. E_bound < 0"""
        vphalf = v + 0.5
        return self.we * vphalf - (self.we * vphalf) ** 2 / (4 * self.De) - self.De

    def intersection_V(self, E):
        arg = (-self.De + np.sqrt(self.De**2 + self.De * E)) / E
        return self.req + np.log(arg) / self.alpha

    def make_rgrid(self, resolution=1000, rmin=None, rmax=None):
        """Make grid of interatomic distances r (Bohr, a.u.)
        - resolution : number of grid points
        """
        if rmin is None:
            rmin = self.rmin
        if rmax is None:
            rmax = self.rmax
        self.r = np.linspace(rmin, rmax, resolution)
        return self.r

    def plot_V(self, ax, **kwargs):
        if not hasattr(self, "r"):
            self.make_rgrid()
        V = self.V(self.r)
        ax.set_xlabel(r"$R$ [a.u.]")
        ax.set_ylabel(r"$E$ [a.u.]")
        ax.plot(self.r, V, **kwargs)


class IntraICEC:
    """ 
    - degeneracyFactor : g_{A^-} / g_A
    - IP: Ionization potential (eV)
    - PI_xs: Function, Fit for Photoionization cross section (eV -> Mb)
    - prefactor: terms that are neither energy nor R dependent 
    """
    def __init__(self, degeneracyFactor: float, IP_A: float, IP_B: float, PI_xs_A: Callable, file_PI_xs_B: str) :
        self.degeneracyFactor = degeneracyFactor # g_A / g_A+
        self.IP_A = IP_A 
        self.IP_B = IP_B # assumption: adiabatic ionization energy
        self.PI_xs_A = PI_xs_A
        self.file_PI_xs_B = file_PI_xs_B
        self.prefactor = (3 * c**2) / (8 * np.pi)

    def define_Morse_B(self, mu, we, req, De):
        """Morse potential for the initial vibrational mode of the system.
        - mu: reduced mass (proton mass)
        - we: Morse parameter (a.u.)
        - req: Equilibrium bond distance (a.u.)
        - De: Dissociation energy (a.u.)
        """
        self.Morse_B = Morse(mu, we, req, De)

    def define_Morse_Bp(self, mu, we, req, De):
        """Morse potential for the initial vibrational mode of the system.
        - mu: reduced mass (proton mass)
        - we: Morse parameter (a.u.)
        - req: Equilibrium bond distance (a.u.)
        - De: Dissociation energy (a.u.)
        """
        self.Morse_Bp = Morse(mu, we, req, De)

    def make_energy_grid(self, minEnergy=0.01*EV2HARTREE, maxEnergy=10*EV2HARTREE, resolution=100, geometric=True): 
        """ Make a suitable grid of incoming electron energies.
        - Energy (a.u.)
        - resolution : number of grid points
        """
        if geometric:
            self.energyGrid = np.geomspace(minEnergy, maxEnergy, resolution)
        else:
            self.energyGrid = np.linspace(minEnergy, maxEnergy, resolution)

    def make_R_grid(self, Rmin=2*ANGSTROM2BOHR, Rmax=10*ANGSTROM2BOHR, resolution=100): 
        """ Make a suitable grid of interatomic distances.
        - R (Bohr, a.u.)
        - resolution : number of grid points
        """
        self.rGrid = np.linspace(Rmin, Rmax, resolution)
    
    def PI_xs_B(self, vi, vf, hbarOmega):
        filename = self.file_PI_xs_B + f"{vi}_{vf}.txt"
        data = np.loadtxt(filename)
        energies, xs = data[:, 0]*EV2HARTREE, data[:, 1]*MB2AU
        interp_func = sp.interpolate.interp1d(energies, xs, kind='linear', fill_value="extrapolate")
        return interp_func(hbarOmega)
    
    def energy_relation(self, electronE, v_B, v_Bp):
        vib_energy_B = 0 if v_Bp is None else (self.Morse_Bp.energy(v_Bp) - self.Morse_Bp.energy(0)) - (self.Morse_B.energy(v_B) - self.Morse_B.energy(0))
        transition_B = self.IP_B - vib_energy_B
        
        hbarOmega = electronE + self.IP_A 
        electronE_f = hbarOmega - transition_B
        
        return hbarOmega, electronE_f
    
    def input_vib_spacing_B(self, vib_spacing_B, vib_spacing_Bp):
        self.vib_diff_to_ground_B = np.cumsum(vib_spacing_B)
        self.vib_diff_to_ground_Bp = np.cumsum(vib_spacing_Bp)

    def energy_relation_vib(self, electronE, v_B, v_Bp):
        hbarOmega = electronE + self.IP_A
        transition_B = self.IP_B + self.vib_diff_to_ground_Bp[v_Bp] - self.vib_diff_to_ground_B[v_B]
        electronE_f = hbarOmega - transition_B
        return hbarOmega, electronE_f

    # ----- CROSS SECTION -----    
    def xs(self, electronE, R, v_B=0, v_Bp=0):
        """ Calculate cross section (a.u.) of ICEC for some kinetic energy and R.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - R: internuclear distance: (Bohr, a.u.)
        - v_A+ -> v_A (v_A -> v_A+ Photoionization)
        - v_B -> v_B+
        """   
        if hasattr(self, 'vib_diff_to_ground_B'):
            hbarOmega, electronE_f = self.energy_relation_vib(electronE, v_B, v_Bp)
        else:
            hbarOmega, electronE_f = self.energy_relation(electronE, v_B, v_Bp)
        if electronE_f <= 0: 
            return 0
        else: 
            # TODO
            PI_xs_A = self.PI_xs_A(hbarOmega)
            PI_xs_B = self.PI_xs_B(v_B, v_Bp, hbarOmega)
            return self.prefactor * self.degeneracyFactor * PI_xs_A * PI_xs_B / (electronE * hbarOmega**2 * R**6)

    def xs_vB_vBp(self, R, v_B=0, v_Bp=0):
        """ Calculate cross section (Mb) of ICEC for given range of kinetic energies.
        - R: internuclear distance: (Bohr, a.u.)
        """        
        xs = np.array([
            self.xs(energy, R, v_B, v_Bp)
            for energy in self.energyGrid
        ]) 
        return xs * AU2MB
    
    def xs_vB(self, R, vB, vBp_max):
        """ Cross section [Mb] for vi -> bound states over range of electron energies.
        """
        # Element-wise summation sum(list_of_arrays)
        xs_array = sum(self.xs_vB_vBp(R, vB, vBp) for vBp in range(vBp_max + 1))
        return xs_array
    
    def spectrum(self, electronE, R, v_B=0, v_Bp_max=0):
        """ Cross sections [Mb] for vi -> bound states given some electron energy.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        """
        spectrum = []
        for v_Bp in range(v_Bp_max+1):
            hbarOmega, electronE_f = self.energy_relation_vib(electronE, v_B, v_Bp)
            if electronE_f >= 0:
                xs = self.xs(electronE, R, v_B, v_Bp)
                spectrum.append([electronE_f * HARTREE2EV, xs * AU2MB, v_Bp])
        return np.array(spectrum)

    def xs_R(self, electronE, v_B=0, v_Bp=None):
        """ Calculate cross section (Mb) of ICEC for given range of interatomic distances.
        - electronE : energy of incoming electron (eV) 
        - R : interatomic distance (Bohr, a.u.)
        """
        electronE = electronE
        if not hasattr(self, 'rGrid'):
            self.make_R_grid()
        xs = np.array([
            self.xs(electronE, r, v_B, v_Bp)
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
            PI_xs = self.PI_xs_A(hbarOmega)
            xs = self.degeneracyFactor * hbarOmega**2 / (2*electronE*c**2) * PI_xs
            PR_xs = np.append(PR_xs, [xs * AU2MB])
        ax.plot(self.energyGrid*HARTREE2EV, PR_xs, **kwargs)
        
        