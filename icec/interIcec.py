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

## Morse Potential
# $$V(R) = D_e \left( 1 - e^{\alpha (R-R_\text{eq})}\right)^2 + E_0$$
# Subtract $D_e$ such that $V(r\to\infty) = 0$ and $V(R_\text{eq}) = - D_e$
# $$V(R) = D_e \left( 1 - e^{\alpha (R-R_\text{eq})}\right)^2 - D_e$$
# Width parameter
# $$\alpha = \omega_0 \sqrt{\frac{\mu}{2 D_e}}$$
# Parameters
# $$\lambda = \frac{\sqrt{2 m D_e}}{\alpha \hbar}$$
# Energy levels for bound states
# $$E_v = w_0 \left(v + \frac{1}{2}\right) - \frac{w_0^2}{4 D_e} \left(v + \frac{1}{2}\right)^2$$
# $$v \in \left\{ n \in \mathbb{N}_0 \, \left| \, n < \lambda - \frac{1}{2} \right.\right\} $$

class Morse:
    """ Initialize the Morse model for a diatomic molecule.
    Adapted from https://scipython.com/blog/the-morse-oscillator and https://liu-group.github.io/Morse-potential
    Input arguments
    - mu: reduced mass (electron mass)
    - we: Morse parameter (cm-1)
    - req: Equilibrium bond distance (Angstrom)
    - De: Dissociation energy (cm-1)
    - E0: Equilibrium energy E(req) (cm-1)
    atomic units (hbar=1, me=1, hartree energy=1)
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
        """ Energy of the v-th (bound) eigenstate of the Morse potential
        E_bound < 0 
        """
        vphalf = v + 0.5
        return self.we * vphalf - (self.we*vphalf)**2 / (4*self.De) - self.De
    
    def intersection_V(self, E):
        arg = (-self.De + np.sqrt(self.De**2 + self.De*E)) / E
        return self.req + np.log(arg)/self.alpha
    
    def define_box(self, box_length = 10*ANGSTROM2BOHR): #lower_bound = None, 
        self.box_length = box_length

    def get_lower_bound(self, E, precision=100):
        ''' Get lower bound for neglecting the diverging r->0 behaviour.
        '''
        R = self.intersection_V(E)
        R_samples = np.linspace(R/2, R, num=precision)
        psi_samples = np.array([    # psi_diss() does not work with np.array due to mpmath
            np.abs(self.psi_diss(E,r)) for r in R_samples
        ])
        min_index = np.argmin(psi_samples)
        print(R_samples[min_index])
        return R_samples[min_index]
    
    def norm_diss(self, E, lower_bound=None):
        ''' Integration over possibly highly-oscillating function
        '''
        if lower_bound is None:
            lower_bound = self.get_lower_bound(E)
        integrand = lambda r: np.conjugate(self.psi_diss(E,r))*self.psi_diss(E,r)
        if E>0.25*EV2HARTREE:
            norm = mpmath.quad(integrand, [lower_bound, self.box_length], maxdegree = 10)
        else:  
            norm = mpmath.quad(integrand, [lower_bound, self.box_length])
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
    
    def psi_diss_2(self, E, r, norm = None):
        """ Dissociative (continuum) states of the Morse potential
        source: https://doi.org/10.1119/1.1485714
        mpmath.hyp1f1: https://mpmath.org/doc/current/functions/hypergeometric.html#hyp1f1
        """
        k = np.sqrt(2*self.mu*E)
        #epsilon = k/self.alpha
        norm = np.sqrt(2*self.mu / np.pi / k)

        r0 = 1 / np.sqrt(2*self.mu*self.De)
        z0 = 2/(self.alpha*r0) * mpmath.exp(self.alpha * self.req)
        #z0 = 2j * self.lam * mpmath.exp(self.alpha * self.req)
        z = z0 * mpmath.exp(-self.alpha * r)

        a_plus  = 0.5 + 1j*k/self.alpha - 1/(self.alpha*r0)
        a_minus = 0.5 - 1j*k/self.alpha - 1/(self.alpha*r0)
        #a_plus  = 0.5 + 1j*k/self.alpha - 1j*self.lam
        #a_minus = 0.5 - 1j*k/self.alpha - 1j*self.lam
        b_plus  = 1 + 2j*k/self.alpha
        b_minus = 1 - 2j*k/self.alpha

        C = cmath.sqrt(mpmath.hyp1f1(a_plus, b_plus, z0) / mpmath.hyp1f1(a_minus, b_minus, z0))
        
        psi_in = 1/C * mpmath.hyp1f1(a_plus, b_plus, z) * mpmath.exp(-1j*k*r)
        psi_out = - C * mpmath.hyp1f1(a_minus, b_minus, z) * mpmath.exp(+1j*k*r)

        psi = norm/(2j) * mpmath.exp(-z/2) * (psi_in + psi_out) 
        return psi

    def make_rgrid(self, resolution=1000, rmin=None, rmax=None):
        """ Make grid of interatomic distances.
        - r: (Bohr, a.u.)
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


## ICEC cross section for atom - atom with internuclear vibrational motion
# Cross section
# \begin{equation}
# \begin{aligned}
#     \sigma (k, i \to f) 
#     &= \frac{3 (\hbar c)^4 }{4 \pi} \,
#     \frac{\sigma_\text{PR}^A(\epsilon_k) \, \sigma_\text{PI}^B(E - \Delta E)}
#         {E^3(E - \Delta E)} \,
#     \left|    
#     \bra{\chi_{AB^+}} R^{-3} \ket{\chi_{A^+B}} 
#     \right|^2 \\
#     &= \frac{3 \hbar^4 c^2 }{8 \pi m_e} \,
#     \frac{g_{A^-}}{g_A} \,
#     \frac{\sigma_\text{PI}^A(E) \, \sigma_\text{PI}^B(E - \Delta E)}
#         {\epsilon_k E(E - \Delta E)} \,
#     \left|    
#     \bra{\chi_{AB^+}} R^{-3} \ket{\chi_{A^+B}} 
#     \right|^2
# \end{aligned}
# \end{equation}
# 
# Transferred energy
# \begin{equation}
# E = \epsilon_k + IP_{A}
# \end{equation}
# 
# Energy into vibrational motion
# \begin{equation}
# \Delta E = E^{AB^+}_{\text{vib},f} - E^{A^+B}_{\text{vib},i}
# \end{equation}
# 
# Threshhold energy
# \begin{equation}
# \begin{aligned}
# \epsilon_t &= IP_A - IP_B - \Delta E\\
# &= IP_A - IP_B + E^{A^+B}_{\text{vib},i} - E^{AB^+}_{\text{vib},f} 
# \end{aligned}
# \end{equation}

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

    def make_energy_grid(self, minEnergy=0, maxEnergy=10, resolution=100): 
        """ Make grid of incoming electron energies.
        - Energy (eV)
        - resolution : number of grid points
        """
        minEnergy = minEnergy * EV2HARTREE
        maxEnergy = maxEnergy * EV2HARTREE
        self.energyGrid = np.arange(minEnergy, maxEnergy, (maxEnergy-minEnergy)/resolution, dtype=float)

    # ===== BOUND - BOUND TRANSITION =====

    def calculate_modified_FC_factor(self, vi, vf):
        """ Calculate <psi_vi|r^-3|psi_vf>
        - vi, vf: initial and final vibrational quantum number
        """
        integrand =  lambda r: np.conjugate(self.Morse_f.psi(vf,r)) * self.Morse_i.psi(vi,r) / r**3
        result, error = sp.integrate.quad(integrand, 0, np.inf)
        return result
    
    def calculate_xs(self, vi, vf, electronE, modifiedFC=None):
        """ Calculate cross section [a.u.] for one vibrational transition.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - modifiedFC : <psi_vi|r^-3|psi_vf>
        """
        if modifiedFC is None:
            modifiedFC = (abs(self.calculate_modified_FC_factor(vi,vf)))**2
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

    def calculate_xs_vivf(self, vi, vf):
        """ Calculate cross section [Mb] for one vibrational transition over range of energies.
        - vi, vf: initial and final vibrational quantum number
        """
        if not hasattr(self, 'energyGrid'):
            self.make_energy_grid()
        modifiedFC = (abs(self.calculate_modified_FC_factor(vi,vf)))**2
        xs_vivf = np.array([
            self.calculate_xs(vi, vf, energy, modifiedFC)
            for energy in self.energyGrid
        ])
        return xs_vivf * AU2MB

    def calculate_xs_vi(self, vi):
        """ Calculate cross section [Mb] for given initial vibrational state over range of kinetic energies.
        Sum over final vibrational states.
        - vi: initial vibrational quantum number
        """   
        #xs_vi = sum(
        #    self.calculate_xs_vivf(vi, vf)
        #    for vf in range(self.Morse_f.vmax + 1)
        #)
        with Pool() as pool:
            result = pool.starmap(
                self.calculate_xs_vivf, zip(repeat(vi), range(self.Morse_f.vmax + 1))
            )
        xs_vi = sum(list(result))
        return xs_vi
    
    def calculate_xs_tot(self):
        """ Calculate cross section [Mb] for range of kinetic energies.
        Sum over final and average over initial vibrational states (should be a Boltzmann average though).
        - self.ICEC_xs_tot : Cross section (Mb)
        """   
        xs = sum(
            self.calculate_xs_vi(vi)
            for vi in range(self.Morse_i.vmax + 1)
        )
        return xs/self.Morse_i.vmax 
    
    def calculate_spectrum(self, electronE, vi=0):
        electronE *= EV2HARTREE
        spectrum = []
        for vf in range(self.Morse_f.vmax + 1):
            deltaE = self.Morse_f.E(vf) - self.Morse_i.E(vi)
            electronE_f = electronE + self.IP_A - self.IP_B - deltaE
            if electronE_f >= 0:
                xs = self.calculate_xs(vi, vf, electronE)
                spectrum.append([electronE_f*HARTREE2EV, xs*AU2MB, vf])
        return np.array(spectrum)

    # ===== BOUND - CONTINUUM TRANSITION =====

    def modified_FC_factor_continuum(self, vi, E, lower_bound = None):
        """ <psi(E)|r^-3|psi_vi>
        """
        #print("modified FC k", np.sqrt(2*self.Morse_f.mu*E))
        if lower_bound is None:
            lower_bound = self.Morse_f.get_lower_bound(E)
        integrand =  lambda r: np.conjugate(self.Morse_f.psi_diss(E,r)) * self.Morse_i.psi(vi,r) / r**3
        result = mpmath.quad(integrand, [lower_bound, self.Morse_f.box_length])
        return result

    def xs_continuum(self, vi, E, electronE, modifiedFC=None, norm=None):
        """ Calculate cross section [a.u.] for one bound-continuum vibrational transition.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - modifiedFC : <psi_vf|r^-3|psi_E>
        """
        if norm is None:
            self.Morse_f.define_box()
            norm = self.Morse_f.norm_diss(E)
        if modifiedFC is None:
            modifiedFC = (np.abs(self.modified_FC_factor_continuum(vi,E)))**2
        deltaE = E - self.Morse_i.E(vi)  # energy that goes into the vibrational transition (Hartree, a.u.)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0 :
            return 0
        else :
            omegaA = electronE + self.IP_A
            PI_xs_A = self.PI_xs_A(omegaA*HARTREE2EV)*MB2AU
            omegaB = omegaA - deltaE
            PI_xs_B = self.PI_xs_B(omegaB*HARTREE2EV)*MB2AU
            xs = self.prefactor * self.degeneracyFactor * PI_xs_A * PI_xs_B * norm * modifiedFC / (electronE * omegaA * omegaB)
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
        norm = self.Morse_f.norm_diss(E, lower_bound=lower_bound)
        modifiedFC = (np.abs(self.modified_FC_factor_continuum(vi,E,lower_bound=lower_bound)))**2
        xs_vi_continuum = np.array([
            self.xs_continuum(vi, E, energy, modifiedFC, norm)
            for energy in self.energyGrid
        ])
        return xs_vi_continuum * AU2MB
    
    def xs_vi_to_all_continuum(self, vi, energies, multi_threaded=True):
        """ Calculate cross section [Mb] for given initial vibrational state over range of kinetic energies.
        Sum over continuum states. Energies of the states are given as input (e.g. find solutions in a box with Mathematica)
        - vi: initial vibrational quantum number
        """
        if multi_threaded:
            with Pool() as pool:
                result = pool.starmap(
                    self.xs_vi_to_continuum, zip(repeat(vi), energies)
                )
            xs_vi = sum(list(result))
        else:
            xs_vi = sum(
                self.xs_vi_to_continuum(vi, E)
                for E in energies
            )

        maxE = (self.energyGrid[-1] + self.IP_A - self.IP_B + self.Morse_i.E(vi))*HARTREE2EV
        print("Maximum vibrational E for highest initial kin.E:", maxE, 'eV' )
        return xs_vi
    
    def spectrum_continuum(self, electronE, energies, vi=0):
        electronE *= EV2HARTREE
        spectrum = []
        for E in energies:
            deltaE = E - self.Morse_i.E(vi)
            electronE_f = electronE + self.IP_A - self.IP_B - deltaE
            if electronE_f >= 0:
                xs = self.xs_continuum(vi, E, electronE)
                spectrum.append([electronE_f*HARTREE2EV, xs*AU2MB, E*HARTREE2EV])
        return np.array(spectrum)

    # ===== OVERLAP CONTRIBUTION =====

    def define_overlap_parameters(self, a_A, a_B, C, d, gaussian_type = 's'):
        self.a_A = a_A
        self.a_B = a_B
        self.C = C
        self.d = d
        self.gaussian_type = gaussian_type

    def calculate_overlap_FC(self, l, electronE, electronE_f, vi, vf):
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
    
    def calculate_overlap_xs(self, vi, vf, electronE, lmax):
        """ Calculate cross section (a.u.) of the overlap contribution.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - lmax: upper bound for the sum over l
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
            else:
                print('Invalid gaussian type')
                return None
            C = self.C * np.exp(-abs(electronE - electronE_f)/self.d)
            sum_l = sum(
                (2*l+1) * abs(self.calculate_overlap_FC(l, electronE, electronE_f, vi, vf))**2
                for l in range(lmax + 1)
            )
            xs = prefactor / electronE**(3/2) / np.sqrt(electronE_f) * sum_l * C
            return xs

    def calculate_overlap_xs_vivf(self, vi, vf, lmax):
        """ Calculate cross section [Mb] of the overlap contribution for one vibrational transition.
        - vi : initial vibrational quantum number
        - vf : final vibrational quantum number
        """
        #prefactor = 32*np.pi * (self.a_A*self.a_B/(self.a_A**2 + self.a_B**2))**3
        overlap_xs_vivf = np.array([
            self.calculate_overlap_xs(vi, vf, energy, lmax)
            for energy in self.energyGrid
        ])
        return overlap_xs_vivf * AU2MB

    def calculate_overlap_xs_vi(self, vi, lmax):
        """ Calculate cross section [Mb] for given initial vibrational state. Sum over final vibrational states.
        - vi: initial vibrational quantum number
        THIS WORKS ONLY IN JUPYTER ON UNIX BASED SYSTEMS
        """  
        with Pool() as pool:
            result = pool.starmap(
                self.calculate_overlap_xs_vivf, zip(repeat(vi), range(self.Morse_f.vmax + 1), repeat(lmax))
            )
        overlap_xs_vi = sum(list(result))
        return overlap_xs_vi
    
    def spectrum_overlap(self, electronE, vi=0):
        electronE *= EV2HARTREE
        spectrum = []
        for vf in range(self.Morse_f.vmax + 1):
            deltaE = self.Morse_f.E(vf) - self.Morse_i.E(vi)
            electronE_f = electronE + self.IP_A - self.IP_B - deltaE
            if electronE_f >= 0:
                xs = self.calculate_overlap_xs(vi, vf, electronE)
                spectrum.append([electronE_f*HARTREE2EV, xs*AU2MB, vf])
        return np.array(spectrum)

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