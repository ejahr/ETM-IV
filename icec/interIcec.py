import numpy as np
import scipy as sp
import cmath
import mpmath
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

# =========================================================
# ==================== Morse Potential ==================== 
# =========================================================
class Morse:
    """ Initialize the Morse model for a diatomic molecule in atomic units (hbar=1, me=1, hartree energy=1).
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
        self.mu = mu # in electron mass
        self.we = we*WAVENUMBER2HARTREE # Hartree
        self.req = req*ANGSTROM2BOHR # Bohr, a.u.
        self.De = De*WAVENUMBER2HARTREE # Hartree
        self.E0 = E0*WAVENUMBER2HARTREE # Hartree
        
        self.alpha = self.we*np.sqrt(self.mu/2/self.De) 
        self.lam = np.sqrt(2 * self.mu * self.De) / self.alpha
        self.z0 = 2 * self.lam * np.exp(self.alpha * self.req)
        self.vmax = int(self.lam - 0.5)
        
        self.rmin = (self.req - np.log(2)/self.alpha)
        f = 0.999
        self.rmax = self.req - np.log(1-f)/self.alpha

    def V(self, r):
        """ Morse potential
        - r : interatomic distance (Bohr, a.u.)
        """
        return self.De * (1 - np.exp(-self.alpha*(r - self.req)))**2 - self.De

    def psi(self, v, r):
        """ v-th eigenstate of the Morse potential
        - r : interatomic distance (Bohr, a.u.)
        """
        z = self.z0 * mpmath.exp(-self.alpha * r)
        # Normalization constant
        Nn = np.sqrt( (2*self.lam-2*v-1) * sp.special.factorial(v) * self.alpha / sp.special.gamma(2*self.lam - v) )
        return Nn * z**(self.lam-v-0.5) * mpmath.exp(-z/2) * mpmath.laguerre(v, 2*self.lam-2*v-1, z)

    def E(self, v):
        """ Energy of the v-th (bound) eigenstate of the Morse potential. E_bound < 0 
        """
        vphalf = v + 0.5
        return self.we * vphalf - (self.we*vphalf)**2 / (4*self.De) - self.De
    
    def intersection_V(self, E):
        arg = (-self.De + np.sqrt(self.De**2 + self.De*E)) / E
        return self.req + np.log(arg)/self.alpha
    
    def define_box(self, box_length = 10*ANGSTROM2BOHR):
        self.box_length = box_length

    def get_lower_bound(self, E, precision=100):
        ''' Get lower bound for neglecting the diverging r->0 behaviour.
        '''
        R = self.intersection_V(E)
        R_samples = np.linspace(R/2, R, num=precision)
        psi_samples = np.array([    # psi_diss() does not work with np.array directly due to mpmath
            np.abs(self.psi_diss(E,r)) for r in R_samples
        ])
        min_index = np.argmin(psi_samples)
        return R_samples[min_index]
    
    def norm_diss(self, E, lower_bound=None):
        ''' Integration over possibly highly-oscillating function
        '''
        if lower_bound is None:
            lower_bound = self.get_lower_bound(E)
        integrand = lambda r: np.conjugate(self.psi_diss(E,r))*self.psi_diss(E,r)
        num_intervals = int(self.box_length/np.pi * np.sqrt(2*self.mu*E) / 10) # approximate oscillating behaviour by particle in a box 
        if num_intervals < 2:
            norm = mpmath.quad(integrand, [lower_bound, self.box_length])
        else:
            intervals = np.linspace(lower_bound, self.box_length, num_intervals + 1)
            norm = 0
            for i in range(num_intervals):
                norm += mpmath.quad(integrand, [intervals[i], intervals[i+1]])
        return 1/mpmath.sqrt(norm)
    
    def psi_diss(self, E, r):
        """ Dissociative (continuum) states of the Morse potential
        source: https://doi.org/10.1088/0953-4075/21/16/011
        mpmath.hyp1f1: https://mpmath.org/doc/current/functions/hypergeometric.html#hyp1f1
        """
        k = np.sqrt(2*self.mu*E)
        epsilon = k/self.alpha
        s = self.lam - 0.5
        z = self.z0 * mpmath.exp(-self.alpha * r) # needs to be mpmath for mpmath.quad to work

        A = mpmath.gamma(-2j*epsilon)/mpmath.gamma(-s-1j*epsilon)
        psi_in = A * z**(1j*epsilon) * mpmath.hyp1f1(-s+1j*epsilon, 2j*epsilon+1, z)
        psi_out = np.conjugate(A) * z**(-1j*epsilon) * mpmath.hyp1f1(-s-1j*epsilon, -2j*epsilon+1, z)

        return mpmath.exp(-z/2) * (psi_in + psi_out) 

    def make_rgrid(self, resolution=1000, rmin=None, rmax=None):
        """ Make grid of interatomic distances r (Bohr, a.u.)
        """
        if rmin is None:
            rmin = self.rmin
        if rmax is None:
            rmax = self.rmax
        self.r = np.linspace(self.rmin, self.rmax, resolution)
        return self.r
    
    def plot_V(self, ax, **kwargs):
        if not hasattr(self, 'r'):
            self.make_rgrid()
        V = self.V(self.r)
        ax.set_xlabel(r'$R$ [a.u.]')
        ax.set_ylabel(r'$E$ [a.u.]')
        ax.plot(self.r, V, **kwargs)

# ==========================================================
# =================== ICEC cross section ===================
# ========== with internuclear vibrational motion ==========
# ==========================================================
class InterICEC:
    """ Calculates the ICEC cross section including nuclear dynamics between the two units.
    - EA: Electron affinity of A (ev)
    - IP: Ionization potential of B (eV)
    - PI_xs_A: Function, Fit for Photoionization cross section of A- (eV -> Mb)
    - PI_xs_B: Function, Fit for Photoionization cross section of B (eV -> Mb)
    - prefactor: terms that are neither energy nor R dependent 
    """
    def __init__(self, degeneracyFactor, IP_A, IP_B, PI_xs_A, PI_xs_B) :
        self.degeneracyFactor = degeneracyFactor
        self.IP_A = IP_A * EV2HARTREE
        self.IP_B = IP_B * EV2HARTREE
        self.PI_xs_A = PI_xs_A
        self.PI_xs_B = PI_xs_B
        self.prefactor = (3 * c**2) / (8 * np.pi)

    def define_Morse_i(self, mu, we, req, De):
        """ Morse potential for the initial vibrational mode of the system.  
        - mu: reduced mass (proton mass)
        - we: Morse parameter (cm-1)
        - req: Equilibrium bond distance (Angstrom)
        - De: Dissociation energy (cm-1)
        """
        self.Morse_i = Morse(mu, we, req, De)

    def define_Morse_f(self, mu, we, req, De):
        """ Morse potential for the initial vibrational mode of the system.  
        - mu: reduced mass (proton mass)
        - we: Morse parameter (cm-1)
        - req: Equilibrium bond distance (Angstrom)
        - De: Dissociation energy (cm-1)
        """
        self.Morse_f = Morse(mu, we, req, De)

    def make_energy_grid(self, minEnergy=0, maxEnergy=10, resolution=100, geometric=True): 
        """ Make grid of incoming electron energies (eV).
        - resolution : number of grid points
        """
        minEnergy = minEnergy * EV2HARTREE
        maxEnergy = maxEnergy * EV2HARTREE
        if geometric:
            self.energyGrid = np.geomspace(minEnergy, maxEnergy, resolution)
        else:
            self.energyGrid = np.arange(minEnergy, maxEnergy, (maxEnergy-minEnergy)/resolution, dtype=float)

    # ===== BOUND - BOUND TRANSITION =====

    def modified_FC_factor(self, vi, vf):
        """ Calculate <psi_vi|r^-3|psi_vf>
        - vi, vf: initial and final vibrational quantum number
        """
        integrand =  lambda r: np.conjugate(self.Morse_f.psi(vf,r)) * self.Morse_i.psi(vi,r) / r**3
        result, error = sp.integrate.quad(integrand, 0, np.inf)
        return result
    
    def xs_bound(self, vi, vf, electronE, modifiedFC=None):
        """ Calculate cross section [a.u.] for one vibrational transition vi -> vf.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - modifiedFC : <psi_vi|r^-3|psi_vf>
        """
        if modifiedFC is None:
            modifiedFC = (abs(self.modified_FC_factor(vi,vf)))**2
        deltaE = self.Morse_f.E(vf) - self.Morse_i.E(vi)   # energy that goes into the vibrational transition (Hartree, a.u.)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0 :
            return 0
        else :
            omegaA = electronE + self.IP_A
            PI_xs_A = self.PI_xs_A(omegaA*HARTREE2EV)*MB2AU
            omegaB = omegaA - deltaE
            PI_xs_B = self.PI_xs_B(omegaB*HARTREE2EV)*MB2AU
            return self.prefactor * self.degeneracyFactor * PI_xs_A * PI_xs_B * modifiedFC / (electronE * omegaA * omegaB)

    def xs_vivf(self, vi, vf):
        """ Calculate cross section [Mb] for one vibrational transition vi -> vf over range of energies.
        """
        if not hasattr(self, 'energyGrid'):
            self.make_energy_grid()
        modifiedFC = (abs(self.modified_FC_factor(vi,vf)))**2
        xs_array = np.array([
            self.xs_bound(vi, vf, energy, modifiedFC)
            for energy in self.energyGrid
        ])
        return xs_array * AU2MB

    def calculate_xs_vi(self, vi):
        """ Calculate cross section [Mb] for given initial vibrational state vi over range of kinetic energies.
        Sum over final vibrational states.
        """   
        with Pool() as pool:
            result = pool.starmap(
                self.xs_vivf, zip(repeat(vi), range(self.Morse_f.vmax + 1))
            )
        xs_array = sum(list(result))
        return xs_array
    
    def calculate_xs_tot(self):
        """ Calculate cross section [Mb] for range of kinetic energies.
        Sum over final and average over initial vibrational states (should be a Boltzmann average though).
        """   
        xs_array = sum(
            self.calculate_xs_vi(vi)
            for vi in range(self.Morse_i.vmax + 1)
        )
        return xs_array/self.Morse_i.vmax 
    
    def calculate_spectrum(self, electronE, vi=0):
        electronE *= EV2HARTREE
        spectrum = []
        for vf in range(self.Morse_f.vmax + 1):
            deltaE = self.Morse_f.E(vf) - self.Morse_i.E(vi)
            electronE_f = electronE + self.IP_A - self.IP_B - deltaE
            if electronE_f >= 0:
                xs = self.xs_bound(vi, vf, electronE)
                spectrum.append([electronE_f*HARTREE2EV, xs*AU2MB, vf])
        return np.array(spectrum)

    # ===== BOUND - CONTINUUM TRANSITION =====

    def modified_FC_continuum(self, vi, E, lower_bound=None, norm=None):
        """ |<psi(E)|r^-3|psi_vi>|^2
        norm: normalization constant for the vibrational continuum state
        divide integration into intervals to deal with the highly oscillating integrand
        """
        if lower_bound is None:
            lower_bound = self.Morse_f.get_lower_bound(E)
        if norm is None:
            norm = self.Morse_f.norm_diss(E)

        integrand =  lambda r: np.conjugate(self.Morse_f.psi_diss(E,r)) * self.Morse_i.psi(vi,r) / r**3
        num_intervals = int(self.Morse_f.box_length/np.pi * np.sqrt(2*self.Morse_f.mu*E) / 10) # approximate oscillating behaviour from particle in a box 
        if num_intervals < 2:
            result = mpmath.quad(integrand, [lower_bound, self.Morse_f.box_length])
        else:
            intervals = np.linspace(lower_bound, self.Morse_f.box_length, num_intervals + 1)
            result = 0
            for i in range(num_intervals):
                result += mpmath.quad(integrand, [intervals[i], intervals[i+1]])

        return (np.abs(norm*result))**2  

    def xs_continuum(self, vi, E, electronE, modifiedFC=None, norm=None):
        """ Calculate cross section [a.u.] for one bound-continuum vibrational transition.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - modifiedFC : |<psi_vf|r^-3|psi_E>|^2
        """
        if modifiedFC is None:
            modifiedFC = self.modified_FC_continuum(vi, E, norm=norm)
        deltaE = E - self.Morse_i.E(vi)  # energy that goes into the vibrational transition (Hartree, a.u.)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0 :
            return 0
        else :
            omegaA = electronE + self.IP_A
            PI_xs_A = self.PI_xs_A(omegaA*HARTREE2EV)*MB2AU
            omegaB = omegaA - deltaE
            PI_xs_B = self.PI_xs_B(omegaB*HARTREE2EV)*MB2AU
            xs = self.prefactor * self.degeneracyFactor * PI_xs_A * PI_xs_B * modifiedFC / (electronE * omegaA * omegaB)
            return np.abs(xs)
    
    def xs_to_all_continuum(self, vi, electronE, energies):
        omegaA = electronE + self.IP_A
        max_energy = omegaA - self.IP_B + self.Morse_i.E(vi)
        xs = sum(
            self.xs_continuum(vi, E) 
            for E in energies if E <= max_energy
        )
        return xs

    def xs_vi_to_continuum(self, vi, E):
        """ Calculate cross section [Mb] for one vibrational transition over range of energies.
        """
        if not hasattr(self, 'energyGrid'):
            self.make_energy_grid()
        if not hasattr(self.Morse_f, 'box_length'):
            self.Morse_f.define_box()
        lower_bound = self.Morse_f.get_lower_bound(E)
        norm = self.Morse_f.norm_diss(E, lower_bound)
        modifiedFC = self.modified_FC_continuum(vi, E, lower_bound, norm)
        xs_array = np.array([
            self.xs_continuum(vi, E, energy, modifiedFC)
            for energy in self.energyGrid
        ])
        return xs_array * AU2MB
    
    def xs_vi_to_all_continuum(self, vi, energies):
        """ Calculate cross section [Mb] for given initial vibrational state vi over range of kinetic energies.
        Sum over continuum states. Energies of the states are given as input (e.g. solutions in a box from Mathematica)
        """
        with Pool() as pool:
            result = pool.starmap(
                self.xs_vi_to_continuum, zip(repeat(vi), energies)
            )
        xs_vi = sum(list(result))

        maxE = (self.energyGrid[-1] + self.IP_A - self.IP_B + self.Morse_i.E(vi))*HARTREE2EV
        print("Maximum vibrational E for highest initial kin.E:", maxE, 'eV' )
        return xs_vi
    
    def function_for_spectrum(self, vi, electronE, E):
        deltaE = E - self.Morse_i.E(vi)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f < 0:
            return [electronE_f*HARTREE2EV, 0, E*HARTREE2EV]
        else:
            xs = self.xs_continuum(vi, E, electronE)
            return electronE_f*HARTREE2EV, xs*AU2MB, E*HARTREE2EV
        
    def spectrum_continuum(self, electronE, energies, vi=0):
        electronE *= EV2HARTREE
        with Pool() as pool:
            result = pool.starmap(
                self.function_for_spectrum, zip(repeat(vi), repeat(electronE), energies)
            )
        return np.array(result)

    # ===== OVERLAP CONTRIBUTION =====
    def define_overlap_parameters(self, a_A, a_B, C, d, lmax = 10, gaussian_type = 's'):
        self.a_A = a_A
        self.a_B = a_B
        self.C = C
        self.d = d
        self.lmax = lmax                        # upper bound for the sum over l
        self.gaussian_type = gaussian_type      # gaussian orbital type for Sab

    # ===== OVERLAP : BOUND - BOUND TRANSITION =====

    def overlap_FC(self, l, electronE, electronE_f, vi, vf):
        """ Calculate <psi_vi|r^-3 exp()|psi_vf>
        - vi, vf: initial and final vibrational quantum number
        """
        a_AB = self.a_A**2 + self.a_B**2
        if self.gaussian_type == 's': 
            integrand = lambda r: (
                np.conjugate(self.Morse_f.psi(vf,r)) * self.Morse_i.psi(vi,r) / r
                * np.exp(-0.5*r**2/a_AB - 0.5*l*(l+1)/(electronE*(self.a_A+r)**2 + electronE_f*(self.a_B+r)**2))
            )
        elif self.gaussian_type == 'pz':
            integrand = lambda r: (
                np.conjugate(self.Morse_f.psi(vf,r)) * self.Morse_i.psi(vi,r)
                * np.exp(-0.5*r**2/a_AB - 0.5*l*(l+1)/(electronE*(self.a_A+r)**2 + electronE_f*(self.a_B+r)**2))
            )
        result, error = sp.integrate.quad(integrand, 0, np.inf)
        return result
    
    def overlap_xs(self, vi, vf, electronE):
        """ Calculate cross section (a.u.) of the overlap contribution.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        """   
        deltaE = self.Morse_f.E(vf) - self.Morse_i.E(vi)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0 :
            return 0
        else:
            if self.gaussian_type == 's':
                prefactor = 4*np.pi * 8*(self.a_A*self.a_B/(self.a_A**2 + self.a_B**2))**3
            elif self.gaussian_type == 'pz':
                prefactor = 4*np.pi * 16*self.a_A**3*(self.a_B/(self.a_A**2 + self.a_B**2))**5
            C = self.C * np.exp(-abs(electronE - electronE_f)/self.d)
            sum_l = sum(
                (2*l+1) * abs(self.overlap_FC(l, electronE, electronE_f, vi, vf))**2
                for l in range(self.lmax + 1)
            )
            xs = prefactor / electronE**(3/2) / np.sqrt(electronE_f) * sum_l * C
            return xs

    def overlap_xs_vivf(self, vi, vf):
        """ Calculate cross section [Mb] of the overlap contribution for one vibrational transition vi -> vf.
        """
        xs_array = np.array([
            self.overlap_xs(vi, vf, energy)
            for energy in self.energyGrid
        ])
        return xs_array * AU2MB

    def overlap_xs_vi(self, vi=0):
        """ Calculate cross section [Mb] for given initial vibrational state vi. Sum over final vibrational states.
        THIS WORKS ONLY IN JUPYTER ON UNIX BASED SYSTEMS
        """  
        with Pool() as pool:
            result = pool.starmap(
                self.overlap_xs_vivf, zip(repeat(vi), range(self.Morse_f.vmax + 1))
            )
        xs_array = sum(list(result))
        return xs_array
    
    def spectrum_overlap(self, electronE, vi=0):
        electronE *= EV2HARTREE
        spectrum = []
        for vf in range(self.Morse_f.vmax + 1):
            deltaE = self.Morse_f.E(vf) - self.Morse_i.E(vi)
            electronE_f = electronE + self.IP_A - self.IP_B - deltaE
            if electronE_f >= 0:
                xs = self.overlap_xs(vi, vf, electronE)
                spectrum.append([electronE_f*HARTREE2EV, xs*AU2MB, vf])
        return np.array(spectrum)
    
    # ===== OVERLAP : BOUND - CONTINUUM TRANSITION =====    

    def overlap_FC_continuum(self, l, electronE, electronE_f, vi, E, lower_bound, norm):
        """ Calculate integral over R-dependent factors of |\int psi_E* psi_vi R^-1 Sab ⟨kf-|ki+⟩ dR|^2
        """
        a_AB = self.a_A**2 + self.a_B**2
        if self.gaussian_type == 's': 
            integrand = lambda r: (
                mpmath.conj(self.Morse_f.psi_diss(E,r)) * self.Morse_i.psi(vi,r) / r
                * mpmath.exp(-0.5*r**2/a_AB - 0.5*l*(l+1)/(electronE*(self.a_A+r)**2 + electronE_f*(self.a_B+r)**2))
            )
        elif self.gaussian_type == 'pz':
            integrand = lambda r: (
                mpmath.conj(self.Morse_f.psi_diss(E,r)) * self.Morse_i.psi(vi,r)
                * mpmath.exp(-0.5*r**2/a_AB - 0.5*l*(l+1)/(electronE*(self.a_A+r)**2 + electronE_f*(self.a_B+r)**2))
            )
        
        num_intervals = int(self.Morse_f.box_length/np.pi * np.sqrt(2*self.Morse_f.mu*E) / 10) # approximate oscillating behaviour by particle in a box 
        if num_intervals < 2:
            result = mpmath.quad(integrand, [lower_bound, self.Morse_f.box_length])
        else:
            intervals = np.linspace(lower_bound, self.Morse_f.box_length, num_intervals + 1)
            result = 0
            for i in range(num_intervals):
                result += mpmath.quad(integrand, [intervals[i], intervals[i+1]])

        return (np.abs(norm*result))**2
    
    def overlap_xs_continuum(self, vi, E, electronE, lower_bound=None, norm=None):
        """ Calculate cross section (a.u.) of the overlap contribution.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        """   
        if lower_bound is None:
            lower_bound = self.Morse_f.get_lower_bound(E)
        if norm is None:
            norm = self.Morse_f.norm_diss(E)

        deltaE = E - self.Morse_i.E(vi)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0 :
            return 0
        else:
            if self.gaussian_type == 's':
                prefactor = 4*np.pi * 8*(self.a_A*self.a_B/(self.a_A**2 + self.a_B**2))**3
            elif self.gaussian_type == 'pz':
                prefactor = 4*np.pi * 16*self.a_A**3*(self.a_B/(self.a_A**2 + self.a_B**2))**5
            else:
                print('Invalid gaussian type')
                return None
            C = self.C * np.exp(-np.abs(electronE - electronE_f)/self.d)
            sum_l = sum(
                (2*l+1) * self.overlap_FC_continuum(l, electronE, electronE_f, vi, E, lower_bound, norm)
                for l in range(self.lmax + 1)
            )
            xs = prefactor / electronE**(3/2) / np.sqrt(electronE_f) * sum_l * C
            return xs
        
    def overlap_xs_vi_to_continuum(self, vi, E):
        """ Calculate cross section [Mb] for one vibrational transition over range of energies.
        """
        lower_bound = self.Morse_f.get_lower_bound(E)
        norm = self.Morse_f.norm_diss(E, lower_bound)
        xs_array = np.array([
            self.overlap_xs_continuum(vi, E, energy, lower_bound, norm)
            for energy in self.energyGrid
        ])
        return xs_array * AU2MB
    
    def overlap_xs_vi_to_all_continuum(self, vi, energies):
        """ Calculate cross section [Mb] for given initial vibrational state vi.
        Sum over continuum states. Energies of the states are given as input (e.g. solutions in a box from Mathematica)
        THIS WORKS ONLY IN JUPYTER ON UNIX BASED SYSTEMS
        """  
        with Pool() as pool:
            result = pool.starmap(
                self.overlap_xs_vi_to_continuum, zip(repeat(vi), energies)
            )
        xs_array = sum(list(result))
        return xs_array
    
    def function_for_overlap_spectrum(self, vi, electronE, E):
        deltaE = E - self.Morse_i.E(vi)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0:
            return [electronE_f*HARTREE2EV, 0, E*HARTREE2EV]
        else:
            xs = self.overlap_xs_continuum(vi, E, electronE)
            return electronE_f*HARTREE2EV, xs*AU2MB, E*HARTREE2EV
        
    def spectrum_overlap_continuum(self, electronE, vi, energies):
        electronE *= EV2HARTREE
        with Pool() as pool:
            result = pool.starmap(
                self.function_for_overlap_spectrum, zip(repeat(vi), repeat(electronE), energies)
            )
        return np.array(result)

    # ===== PLOTTING FUNCTIONS =====

    def plot_xs(self, ax, ICEC_xs, label="ICEC", **kwargs):
        """ Plot the cross section [Mb] for a given vibrational transition w.r.t. the kinetic energy of the incoming electron.
        """
        ax.plot(self.energyGrid*HARTREE2EV, ICEC_xs, label=label, **kwargs)
        ax.set_xlabel(r'$E_\text{el}$ [eV]')
        ax.set_ylabel(r'$\sigma$ [Mb]')
        ax.set_yscale('log')
        ax.set_title('ICEC cross section')

    def plot_PR_xs(self, ax, label="PR", linestyle='dashed', **kwargs):
        """ Plot the photorecombination cross section [a.u.] w.r.t. the kinetic energy of the electron.
        """
        PR_xs = np.array([])
        for electronE in self.energyGrid:
            hbarOmega = electronE + self.IP_A
            PI_xs = self.PI_xs_A(hbarOmega*HARTREE2EV)*MB2AU
            xs = self.degeneracyFactor * hbarOmega**2 / (2*electronE*c**2) * PI_xs
            PR_xs = np.append(PR_xs, [xs * AU2MB])
        ax.plot(self.energyGrid*HARTREE2EV, PR_xs, label=label, linestyle=linestyle, **kwargs)