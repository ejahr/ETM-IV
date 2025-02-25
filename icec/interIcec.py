import numpy as np
import scipy as sp
import mpmath
import copy
import matplotlib.pyplot as plt
from itertools import repeat
from multiprocessing import Pool
from crosssection.icec.constants import *

# =========================================================
# ==================== Morse Potential ====================
# =========================================================
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
        self.we = we * WAVENUMBER2HARTREE  # Hartree
        self.req = req * ANGSTROM2BOHR  # Bohr, a.u.
        self.De = De * WAVENUMBER2HARTREE  # Hartree
        self.E0 = E0 * WAVENUMBER2HARTREE  # Hartree

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
    
    def dVdr(self, r):
        return 2 * self.alpha * self.De * np.exp(-self.alpha * (r - self.req)) * (1 - np.exp(-self.alpha * (r - self.req))) 

    def psi(self, v, r):
        """v-th eigenstate of the Morse potential
        - r : interatomic distance (Bohr, a.u.)
        """
        z = self.z0 * mpmath.exp(-self.alpha * r)
        N = mpmath.sqrt(
            (2 * self.lam - 2 * v - 1)
            * mpmath.factorial(v)
            * self.alpha
            / mpmath.gamma(2 * self.lam - v)
        )
        return (
            N
            * z ** (self.lam - v - 0.5)
            * mpmath.exp(-z / 2)
            * mpmath.laguerre(v, 2 * self.lam - 2 * v - 1, z)
        )

    def E(self, v):
        """Energy [Hartree] of the v-th (bound) Morse state. E_bound < 0"""
        vphalf = v + 0.5
        return self.we * vphalf - (self.we * vphalf) ** 2 / (4 * self.De) - self.De

    def intersection_V(self, E):
        arg = (-self.De + np.sqrt(self.De**2 + self.De * E)) / E
        return self.req + np.log(arg) / self.alpha
    
    def reflection_point(self, E):
        arg = 1 + np.sqrt((E + self.De)/self.De)
        return self.req - np.log(arg)/self.alpha

    def define_box(self, box_length=10 * ANGSTROM2BOHR):
        self.box_length = box_length

    def get_lower_bound(self, E, precision=200):
        """ Lower bound for neglecting the diverging r->0 behaviour of the dissociative Morse states. """
        R = self.intersection_V(E)
        R_samples = np.linspace(R / 2, R, num=precision)
        psi_samples = np.array(
            [  # psi_diss() does not work with np.array directly due to mpmath
                np.abs(self.psi_diss(E, r)) for r in R_samples
            ]
        )
        min_index = np.nanargmin(psi_samples)
        return R_samples[min_index]

    def estimate_oscillation(self, E, d=5):
        # particle in a box: E_n = n^2*pi^2/(2*m*L^2)
        n = self.box_length * np.sqrt(2 * self.mu * E) / np.pi
        return round(n / d)  # divide by d to not have just one period per interval

    def norm_diss(self, E, lower_bound=None):
        """ Box normalization of the dissociative Morse states.
        These states can be highly-oscillating 
        """
        def integrand(r):
            return mpmath.conj(self.psi_diss(E, r)) * self.psi_diss(E, r)
        if lower_bound is None:
            lower_bound = self.get_lower_bound(E)
        r_reflection = self.reflection_point(E)
        num_intervals = self.estimate_oscillation(E)
        if num_intervals < 10:
            norm = mpmath.quadsubdiv(integrand, [lower_bound, r_reflection, self.rmax, self.box_length], maxdegree=10)
        else:
            norm = mpmath.quadsubdiv(integrand, [lower_bound, r_reflection], maxdegree=10)
            intervals_mid = np.linspace(r_reflection, self.rmax, num_intervals + 1)
            norm += mpmath.quadsubdiv(integrand, intervals_mid, maxdegree=10)
            intervals_high = np.linspace(self.rmax, self.box_length, num_intervals + 1)
            norm += mpmath.quadsubdiv(integrand, intervals_high, maxdegree=10)
        return 1 / mpmath.sqrt(norm)

    def psi_diss(self, E, r):
        """ Dissociative (continuum) states of the Morse potential
        source: https://doi.org/10.1088/0953-4075/21/16/011
        mpmath.hyp1f1: https://mpmath.org/doc/current/functions/hypergeometric.html#hyp1f1
        """
        k = mpmath.sqrt(2 * self.mu * E)
        epsilon = k / self.alpha
        s = self.lam - 0.5
        z = self.z0 * mpmath.exp(-self.alpha * r)  
        A = mpmath.gamma(-2j * epsilon) / mpmath.gamma(-s - 1j * epsilon)
        psi_in = (
            A
            * z ** (1j * epsilon)
            * mpmath.hyp1f1(-s + 1j * epsilon, 2j * epsilon + 1, z)
        )
        psi_out = (
            mpmath.conj(A)
            * z ** (-1j * epsilon)
            * mpmath.hyp1f1(-s - 1j * epsilon, -2j * epsilon + 1, z)
        )
        return mpmath.exp(-z / 2) * (psi_in + psi_out)

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


# ==========================================================
# =================== ICEC cross section ===================
# ========== with internuclear vibrational motion ==========
# ==========================================================
class InterICEC:
    """Calculates the ICEC cross section including nuclear dynamics between the two units.
    - EA: Electron affinity of A (ev)
    - IP: Ionization potential of B (eV)
    - PI_xs_A: Function, Fit for Photoionization cross section of A- (eV -> Mb)
    - PI_xs_B: Function, Fit for Photoionization cross section of B (eV -> Mb)
    - prefactor: terms that are neither energy nor R dependent
    """

    def __init__(self, degeneracyFactor, IP_A, IP_B, PI_xs_A, PI_xs_B):
        self.degeneracyFactor = degeneracyFactor
        self.IP_A = IP_A * EV2HARTREE
        self.IP_B = IP_B * EV2HARTREE
        self.PI_xs_A = PI_xs_A
        self.PI_xs_B = PI_xs_B
        self.prefactor = (3 * c**2) / (8 * np.pi)

    def define_Morse_i(self, mu, we, req, De):
        """Morse potential for the initial vibrational mode of the system.
        - mu: reduced mass (proton mass)
        - we: Morse parameter (cm-1)
        - req: Equilibrium bond distance (Angstrom)
        - De: Dissociation energy (cm-1)
        """
        self.Morse_i = Morse(mu, we, req, De)

    def define_Morse_f(self, mu, we, req, De):
        """Morse potential for the initial vibrational mode of the system.
        - mu: reduced mass (proton mass)
        - we: Morse parameter (cm-1)
        - req: Equilibrium bond distance (Angstrom)
        - De: Dissociation energy (cm-1)
        """
        self.Morse_f = Morse(mu, we, req, De)

    def make_energy_grid(
        self, minEnergy=0, maxEnergy=10, resolution=100, geometric=True
    ):
        """Make grid of incoming electron energies (Hartree).
        - minEnergy / maxEnergy [eV]
        - resolution : number of grid points
        """
        minEnergy = minEnergy * EV2HARTREE
        maxEnergy = maxEnergy * EV2HARTREE
        if geometric:
            self.energyGrid = np.geomspace(minEnergy, maxEnergy, resolution)
        else:
            self.energyGrid = np.arange(
                minEnergy, maxEnergy, (maxEnergy - minEnergy) / resolution, dtype=float
            )

    # ===== BOUND - BOUND TRANSITION =====

    def modified_FC_factor(self, vi, vf):
        """<psi_vi|r^-3|psi_vf>"""
        def integrand(r):
            return np.conjugate(self.Morse_f.psi(vf, r)) * self.Morse_i.psi(vi, r) / r ** 3
        result, error = sp.integrate.quad(integrand, 0, np.inf)
        return result

    def xs_bb(self, vi, vf, electronE, modifiedFC=None):
        """ Cross section [a.u.] for vi -> vf given some electron energy.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        - modifiedFC : <psi_vi|r^-3|psi_vf>
        """
        if modifiedFC is None:
            modifiedFC = (abs(self.modified_FC_factor(vi, vf))) ** 2
        # energy that goes into the vibrational transition (Hartree, a.u.)
        deltaE = self.Morse_f.E(vf) - self.Morse_i.E(vi)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0:
            return 0
        else:
            omegaA = electronE + self.IP_A
            PI_xs_A = self.PI_xs_A(omegaA * HARTREE2EV) * MB2AU
            omegaB = omegaA - deltaE
            PI_xs_B = self.PI_xs_B(omegaB * HARTREE2EV) * MB2AU
            return (
                self.prefactor
                * self.degeneracyFactor
                * PI_xs_A
                * PI_xs_B
                * modifiedFC
                / (electronE * omegaA * omegaB)
            )

    def xs_vivf(self, vi, vf):
        """ Cross section [Mb] for vi -> vf over range of electron energies."""
        if not hasattr(self, "energyGrid"):
            self.make_energy_grid()
        modifiedFC = (abs(self.modified_FC_factor(vi, vf))) ** 2
        xs_array = np.array(
            [self.xs_bb(vi, vf, energy, modifiedFC) for energy in self.energyGrid]
        )
        return xs_array * AU2MB

    def xs_vi(self, vi):
        """ Cross section [Mb] for vi -> bound states over range of electron energies.
        POOL WORKS ONLY IN JUPYTER ON UNIX BASED SYSTEMS
        """
        with Pool() as pool:
            result = pool.starmap(
                self.xs_vivf, zip(repeat(vi), range(self.Morse_f.vmax + 1))
            )
        # Element-wise summation sum(list_of_arrays), see https://stackoverflow.com/questions/66111665/numpy-element-wise-addition-with-multiple-arrays
        xs_array = sum(list(result)) 
        return xs_array

    def xs_tot(self):
        """ Cross section [Mb] over range of electron energies.
        Sum over final and average over initial vibrational states (should be a Boltzmann average though).
        """
        xs_array = sum(self.xs_vi(vi) for vi in range(self.Morse_i.vmax + 1))
        return xs_array / self.Morse_i.vmax

    def spectrum_bb(self, electronE, vi=0):
        """ Cross sections [Mb] for vi -> bound states given some electron energy.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        """
        electronE *= EV2HARTREE
        spectrum = []
        for vf in range(self.Morse_f.vmax + 1):
            deltaE = self.Morse_f.E(vf) - self.Morse_i.E(vi)
            electronE_f = electronE + self.IP_A - self.IP_B - deltaE
            if electronE_f >= 0:
                xs = self.xs_bb(vi, vf, electronE)
                spectrum.append([electronE_f * HARTREE2EV, xs * AU2MB, vf])
        return np.array(spectrum)

    # ===== BOUND - CONTINUUM TRANSITION =====
    
    def FC_integrand(self, vi, E, r):
        return mpmath.conj(self.Morse_f.psi_diss(E, r)) * self.Morse_i.psi(vi, r) / r ** 3
    
    def integrate_r(self, integrand, vi, E, lower_bound=None):
        if lower_bound is None:
            lower_bound = self.Morse_f.get_lower_bound(E)
        r_reflection = self.Morse_f.reflection_point(E)
        rmax = max(self.Morse_i.rmax, self.Morse_f.rmax)

        num_intervals = self.Morse_f.estimate_oscillation(E)
        if num_intervals < 10:
            return mpmath.quadsubdiv(integrand, [lower_bound, r_reflection, rmax, self.Morse_f.box_length], maxdegree=10)
        else:
            result = mpmath.quadsubdiv(integrand, [lower_bound, r_reflection], maxdegree=10)
            factor = (max(1, vi + 4 - self.Morse_i.vmax))**2
            intervals_mid = np.linspace(r_reflection, rmax, factor*num_intervals+1)
            result += mpmath.quadsubdiv(integrand, intervals_mid, maxdegree=10)
            intervals_high = np.linspace(rmax, self.Morse_f.box_length, num_intervals+1)
            result += mpmath.quadsubdiv(integrand, intervals_high, maxdegree=10)
            return result

    def FC_continuum(self, vi, E, lower_bound=None, norm=None):
        """|<psi(E)|r^-3|psi_vi>|^2
        norm: normalization constant for the vibrational continuum state
        divide integration into intervals to deal with highly oscillating integrand
        """
        if norm is None:
            norm = self.Morse_f.norm_diss(E)

        def integrand(r):
            return mpmath.conj(self.Morse_f.psi_diss(E, r)) * self.Morse_i.psi(vi, r) / r ** 3
        
        result = self.integrate_r(integrand, vi, E, lower_bound=lower_bound)    

        return (mpmath.fabs(norm * result)) ** 2

    def function_for_FC(self, vi, E, electronE, density_of_states_at_E):
        deltaE = E - self.Morse_i.E(vi)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0:
            return electronE_f * HARTREE2EV, 0, E * HARTREE2EV
        else:
            modifiedFC = self.FC_continuum(vi, E) * density_of_states_at_E
            return electronE_f*HARTREE2EV, modifiedFC, E*HARTREE2EV # a.u.
        
    def spectrum_bc_FC(self, electronE, vi, diss_energies):
        electronE *= EV2HARTREE
        density_of_states = self.get_density_of_states(diss_energies)
        with Pool() as pool:
            result = pool.starmap(
                self.function_for_FC, 
                zip(repeat(vi), diss_energies, repeat(electronE), density_of_states)
            )
        return np.array(result)

    def xs_bc(self, vi, E, electronE, modifiedFC=None, norm=None):
        """Cross section [a.u.] for one bound-continuum vibrational transition vi -> E.
        - E [Hartree] : energy of the dissociative Morse state
        - electronE [Hartree] : kinetic energy of incoming electron
        - modifiedFC [a.u.] : |<psi_vf|r^-3|psi_E>|^2
        """
        if modifiedFC is None:
            modifiedFC = self.FC_continuum(vi, E, norm=norm)
        # energy that goes into the vibrational transition (Hartree, a.u.)
        deltaE = E - self.Morse_i.E(vi)  
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0:
            return 0
        else:
            omegaA = electronE + self.IP_A
            PI_xs_A = self.PI_xs_A(omegaA * HARTREE2EV) * MB2AU
            omegaB = omegaA - deltaE
            PI_xs_B = self.PI_xs_B(omegaB * HARTREE2EV) * MB2AU
            xs = (
                self.prefactor
                * self.degeneracyFactor
                * PI_xs_A
                * PI_xs_B
                * modifiedFC
                / (electronE * omegaA * omegaB)
            )
            return float(xs)

    def xs_to_continuum(self, vi, diss_energies, electronE):
        """Cross section [a.u.] for vi -> continuum states.
        - diss_energies [Hartree] : energies of all possible dissociative states (in a box)
        - electronE [Hartree] : kinetic energy of incoming electron
        """
        omegaA = electronE + self.IP_A
        max_energy = omegaA - self.IP_B + self.Morse_i.E(vi)
        xs = sum(self.xs_bc(vi, E) for E in diss_energies if E <= max_energy)
        return xs

    def xs_vi_to_E(self, vi, E):
        """Cross section [Mb] for vi -> E over range of electron energies."""
        if not hasattr(self, "energyGrid"):
            self.make_energy_grid()
        if not hasattr(self.Morse_f, "box_length"):
            self.Morse_f.define_box()
        lower_bound = self.Morse_f.get_lower_bound(E)
        norm = self.Morse_f.norm_diss(E, lower_bound)
        modifiedFC = self.FC_continuum(vi, E, lower_bound, norm)
        xs_array = np.array(
            [self.xs_bc(vi, E, energy, modifiedFC) for energy in self.energyGrid]
        )
        return xs_array * AU2MB

    def xs_vi_to_continuum(self, vi, diss_energies):
        """Cross section [Mb] for vi -> continuum over range of electron energies.
        - diss_energies [Hartree] : energies of all possible dissociative states (in a box)
        POOL WORKS ONLY IN JUPYTER ON UNIX BASED SYSTEMS
        """
        with Pool() as pool:
            result = pool.starmap(
                self.xs_vi_to_E, zip(repeat(vi), diss_energies)
            )
        xs_array = sum(list(result))
        # maxE = (self.energyGrid[-1] + self.IP_A - self.IP_B + self.Morse_i.E(vi))*HARTREE2EV
        # print("Maximum vibrational E for highest initial kin.E:", maxE, 'eV' )
        return xs_array
    
    def get_density_of_states(self, diss_energies):
        density_of_states = np.zeros(len(diss_energies))
        density_of_states[1:-1] = 2/(diss_energies[2:] - diss_energies[0:-2])
        density_of_states[0] = 1/(diss_energies[1]-diss_energies[0])
        density_of_states[-1] = 1/(diss_energies[-1]-diss_energies[-2])
        return density_of_states

    def function_for_spectrum(self, electronE, vi, E, density_of_states_at_E):
        deltaE = E - self.Morse_i.E(vi)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0:
            return electronE_f * HARTREE2EV, 0, E * HARTREE2EV
        else:
            # transform to energy normalization by multiplying with the density of states at E
            xs = self.xs_bc(vi, E, electronE) * density_of_states_at_E
            return electronE_f * HARTREE2EV, xs * AU2MB,  E * HARTREE2EV

    def spectrum_bc(self, electronE, vi, diss_energies):
        """ Cross sections [Mb] for vi -> continuum given a single electron energy.
        - electronE [Hartree] : kinetic energy of incoming electron 
        - diss_energies [Hartree] : energies of all possible dissociative states (in a box)
        """
        electronE *= EV2HARTREE
        density_of_states = self.get_density_of_states(diss_energies)
        with Pool() as pool:
            result = pool.starmap(
                self.function_for_spectrum, 
                zip(repeat(electronE), repeat(vi), diss_energies, density_of_states)
            )
        return np.array(result)

    # ===== PLOTTING FUNCTIONS =====

    def plot_xs(self, ax, ICEC_xs, label="ICEC", **kwargs):
        """Plots the cross section [Mb] for a given vibrational transition w.r.t. the energy of the incoming electron."""
        ax.plot(self.energyGrid * HARTREE2EV, ICEC_xs, label=label, **kwargs)
        ax.set_xlabel(r"$E_\text{el}$ [eV]")
        ax.set_ylabel(r"$\sigma$ [Mb]")
        ax.set_yscale("log")
        ax.set_title("ICEC cross section")

    def plot_PR_xs(self, ax, label="PR", linestyle="dashed", **kwargs):
        """Plots the photorecombination cross section [a.u.] w.r.t. the energy of the electron."""
        PR_xs = np.array([])
        for electronE in self.energyGrid:
            hbarOmega = electronE + self.IP_A
            PI_xs = self.PI_xs_A(hbarOmega * HARTREE2EV) * MB2AU
            xs = self.degeneracyFactor * hbarOmega**2 / (2 * electronE * c**2) * PI_xs
            PR_xs = np.append(PR_xs, [xs * AU2MB])
        ax.plot(
            self.energyGrid * HARTREE2EV,
            PR_xs,
            label=label,
            linestyle=linestyle,
            **kwargs,
        )

# ==========================================================
# =================== ICEC cross section ===================
# ==== for the Overlap (electron transfer) contribution ====
# ========== with internuclear vibrational motion ==========
# ==========================================================
class OverlapInterICEC(InterICEC):
        
    @classmethod
    def from_InterICEC(cls, InstanceICEC: InterICEC):
        ''' generate an instance of OverlapInterICEC from an instance of InterICEC 
        https://stackoverflow.com/questions/71209560/initialize-a-superclass-with-an-existing-object-copy-constructor
        '''
        new_inst = copy.deepcopy(InstanceICEC) 
        new_inst.__class__ = cls
        return new_inst
    
    def define_overlap_parameters(self, a_A, a_B, C, d, lmax=10, gaussian_type="s"):
        self.a_A = a_A
        self.a_B = a_B
        self.C = C
        self.d = d
        self.lmax = lmax  # upper bound for the partial wave expansion
        self.gaussian_type = gaussian_type  # gaussian orbital type for Sab

    # ===== BOUND - BOUND TRANSITION =====
    def modified_FC(self, l, electronE, electronE_f, vi, vf):
        """ <psi_vi|r^-3 exp()|psi_vf>
        - l : total angular momentum of the partial wave
        """
        a_AB = self.a_A**2 + self.a_B**2
        if self.gaussian_type == "s":
            def integrand(r):
                return np.conjugate(self.Morse_f.psi(vf, r)) * self.Morse_i.psi(vi, r) / r \
                    * np.exp(-0.5 * r ** 2 / a_AB - 0.5 * l * (l + 1) 
                             / (electronE * (self.a_A + r) ** 2 + electronE_f * (self.a_B + r) ** 2))
        elif self.gaussian_type == "pz":
            def integrand(r):
                return np.conjugate(self.Morse_f.psi(vf, r)) * self.Morse_i.psi(vi, r) \
                    * np.exp(-0.5 * r ** 2 / a_AB - 0.5 * l * (l + 1) 
                             / (electronE * (self.a_A + r) ** 2 + electronE_f * (self.a_B + r) ** 2))
        result, error = sp.integrate.quad(integrand, 0, np.inf)
        return result

    def xs_bb(self, vi, vf, electronE):
        """Cross section [a.u.] of the overlap contribution.
        - electronE [Hartree] : kinetic energy of incoming electron
        """
        deltaE = self.Morse_f.E(vf) - self.Morse_i.E(vi)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0:
            return 0
        else:
            if self.gaussian_type == "s":
                prefactor = (
                    4
                    * np.pi
                    * 8
                    * (self.a_A * self.a_B / (self.a_A**2 + self.a_B**2)) ** 3
                )
            elif self.gaussian_type == "pz":
                prefactor = (
                    4
                    * np.pi
                    * 16
                    * self.a_A**3
                    * (self.a_B / (self.a_A**2 + self.a_B**2)) ** 5
                )
            C = self.C * np.exp(-abs(electronE - electronE_f) / self.d)
            sum_l = sum(
                (2 * l + 1)
                * abs(self.modified_FC(l, electronE, electronE_f, vi, vf)) ** 2
                for l in range(self.lmax + 1)
            )
            xs = prefactor / electronE ** (3 / 2) / np.sqrt(electronE_f) * sum_l * C
            return xs

    def xs_vivf(self, vi, vf):
        """Cross section [Mb] of the overlap contribution for vi -> vf over range of electron energies."""
        xs_array = np.array(
            [self.xs_bb(vi, vf, energy) for energy in self.energyGrid]
        )
        return xs_array * AU2MB

    # ===== BOUND - CONTINUUM TRANSITION =====
    
    def FC_integrand(self, vi, E, electronE, electronE_f, l, r):
        a_AB = self.a_A**2 + self.a_B**2
        return mpmath.conj(self.Morse_f.psi_diss(E, r)) * self.Morse_i.psi(vi, r) / r \
            * mpmath.exp(-0.5 * r ** 2 / a_AB - 0.5 * l * (l + 1) / (electronE * (self.a_A + r) ** 2 + electronE_f * (self.a_B + r) ** 2))
    
    def FC_continuum(self, vi, E, electronE, electronE_f, l, lower_bound, norm):
        """Integral over R-dependent factors of |\int psi_E* psi_vi R^-1 Sab ⟨kf-|ki+⟩ dR|^2
        - l : total angular momentum of the partial wave
        """
        a_AB = self.a_A**2 + self.a_B**2
        if self.gaussian_type == "s":
            def integrand(r):
                return mpmath.conj(self.Morse_f.psi_diss(E, r)) * self.Morse_i.psi(vi, r) / r \
                    * mpmath.exp(-0.5 * r ** 2 / a_AB - 0.5 * l * (l + 1) / (electronE * (self.a_A + r) ** 2 + electronE_f * (self.a_B + r) ** 2))
        elif self.gaussian_type == "pz":
            def integrand(r):
                return mpmath.conj(self.Morse_f.psi_diss(E, r)) * self.Morse_i.psi(vi, r) \
                    * mpmath.exp(-0.5 * r ** 2 / a_AB - 0.5 * l * (l + 1) / (electronE * (self.a_A + r) ** 2 + electronE_f * (self.a_B + r) ** 2))

        result = self.integrate_r(integrand, vi, E, lower_bound=lower_bound) 

        return (mpmath.fabs(norm * result)) ** 2

    def xs_bc(self, vi, E, electronE, lower_bound=None, norm=None):
        """Cross section (a.u.) of the overlap contribution.
        - electronE : kinetic energy of incoming electron (Hartree, a.u.)
        """
        if lower_bound is None:
            lower_bound = self.Morse_f.get_lower_bound(E)
        if norm is None:
            norm = self.Morse_f.norm_diss(E)

        deltaE = E - self.Morse_i.E(vi)
        electronE_f = electronE + self.IP_A - self.IP_B - deltaE
        if electronE_f <= 0:
            return 0
        else:
            if self.gaussian_type == "s":
                prefactor = (
                    4 * np.pi * 8 * (self.a_A * self.a_B / (self.a_A**2 + self.a_B**2)) ** 3
                )
            elif self.gaussian_type == "pz":
                prefactor = (
                    4 * np.pi * 16 * self.a_A**3 * (self.a_B / (self.a_A**2 + self.a_B**2)) ** 5
                )
            else:
                print("Invalid gaussian type")
                return None
            C = self.C * mpmath.exp(-mpmath.fabs(electronE - electronE_f) / self.d)
            sum_l = sum(
                (2 * l + 1) * self.FC_continuum(vi, E, electronE, electronE_f, l, lower_bound, norm)
                for l in range(self.lmax + 1)
            )
            xs = prefactor / electronE**(3 / 2) / mpmath.sqrt(electronE_f) * sum_l * C
            # xs = prefactor / electronE**(3/2) * sum_l * C # TEST
            return np.abs(xs)

    def xs_vi_to_E(self, vi, E):
        """Overlap cross section [Mb] for vi -> E over range of electron energies."""
        lower_bound = self.Morse_f.get_lower_bound(E)
        norm = self.Morse_f.norm_diss(E, lower_bound)
        xs_array = np.array(
            [ self.xs_bc(vi, E, energy, lower_bound, norm) for energy in self.energyGrid ]
        )
        return xs_array * AU2MB