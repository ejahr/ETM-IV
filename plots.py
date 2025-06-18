import os
import glob
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from HeNe_input import *  # defines constants and parameters

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams.update({"font.size": 18})

# path to the folder containing a folder named 'results' containing the results files
DIRECTORY = "./"

# energy
EV2HARTREE = 3.67493e-2
HARTREE2EV = 27.2114
WAVENUMBER2HARTREE = 4.55633e-6 # cm^-1
J2HARTREE = 2.2937104486906e17

# length
BOHR2M = 5.29177210544e-11
M2BOHR = 18897259885.789
ANGSTROM2BOHR = 1e-10 * M2BOHR
BOHR2ANGSTROM = 0.529177249

# cross section
MB2M2 = 1e-22
MB2AU = MB2M2 * M2BOHR**2
AU2MB = BOHR2M**2 / MB2M2

# physical constants in a.u.
c = 137
KB = 1.380649e-23 * J2HARTREE # Hartree/K


class RMatrixProcessor:
    """Plots ICEC cross sections from files calculated using the R-matrix approach.
    - process : identifier for the transition / process to get correct data
    - color : defines standard color for rmatrix lines
    - summed : if b-b and b-d contributions should be summed up"""

    def __init__(self, process, color="tab:blue", summed: bool = False):
        self.process = process
        self.color = color
        self.summed = summed

    @staticmethod
    def read_file(fname):
        with open(fname) as f:
            sections = [
                np.loadtxt(section.splitlines(), comments="#")
                for section in f.read().strip().split("\n\n")
                if section.strip()
            ]
        return np.array(sections, dtype=object)

    @staticmethod
    def interpolate(energy1, energy2, xs2):
        interp_xs2 = sp.interpolate.interp1d(
            energy2, xs2, kind="linear", bounds_error=False, fill_value=0
        )
        return interp_xs2(energy1)

    def get_constR(self, shift=0):
        fname = DIRECTORY + "r-matrix/R-matrix.R_eq." + self.process
        rmatrix = np.loadtxt(fname, comments="#")
        energies, xs = rmatrix[:, 0], rmatrix[:, 1] * AU2MB
        energies += shift
        mask = energies > 0
        return np.array(energies[mask], dtype=float), np.array(xs[mask], dtype=float)

    def get_bb(self, vi):
        fname = DIRECTORY + "r-matrix/b-b.av.out." + self.process
        rmatrix = self.read_file(fname)
        energies, xs = rmatrix[vi][:, 0], rmatrix[vi][:, 1] * AU2MB
        mask = energies > 0
        return np.array(energies[mask], dtype=float), np.array(xs[mask], dtype=float)

    def get_bc(self, vi):
        file_list = glob.glob(DIRECTORY + "r-matrix/b-c.av.out." + self.process + "*")
        if file_list:
            rmatrix = self.read_file(file_list[0])
            energies, xs = rmatrix[vi][:, 0], rmatrix[vi][:, 1] * AU2MB
            mask = energies > 0
            return np.array(energies[mask], dtype=float), np.array(
                xs[mask], dtype=float
            )
        else:
            print("No matching files found.")
            return None, None

    def plot_constR(self, ax, shift=0, **kwargs):
        energies, xs = self.get_constR(shift)
        ax.plot(
            energies, xs, color=self.color, linestyle=":", **kwargs
        )  # , label=r"$R_\text{e}$ R-matrix"

    def plot_xs(self, ax: plt.Axes, vi, **kwargs):
        """Plots ICEC cross section corresponding to bound-bound and bound-dissociative (or continuum) transitions of the dimer."""
        energies_bb, xs_bb = self.get_bb(vi)
        energies_bc, xs_bc = self.get_bc(vi)
        if energies_bc is None or xs_bc is None:
            return  # Skip plotting if no b-d data
        if not self.summed:
            ax.plot(
                energies_bb, xs_bb, color=self.color, dashes=(5, 3), **kwargs
            )  # , label=r"b-b R-matrix"
            ax.plot(energies_bc, xs_bc, color=self.color, label=r"R-matrix", **kwargs)  # b-d
        else:
            xs_bc = self.interpolate(energies_bb, energies_bc, xs_bc, **kwargs)
            xs = xs_bb + xs_bc
            ax.plot(energies_bb, xs, color=self.color, label=r"R-matrix", **kwargs)

    def boltzmann_factor(self, we, De, v, T):
        E_morse = we * (v + 0.5) - (we * (v + 0.5)) ** 2 / (4 * De) - De
        return np.exp(-E_morse / KB / T)

    def plot_xs_temperature(self, ax: plt.Axes, T, we, De, vmax, color="tab:blue"):
        """Calculates and plots the temperature dependent ICEC cross sections."""
        norm = 1.0 / sum(
            self.boltzmann_factor(we, De, vi, T) for vi in range(vmax)
        )  # ignore last initial vib state

        vi = 0
        energies, xs_bb = self.get_bb(vi)
        avg_bb = self.boltzmann_factor(we, De, vi, T) * xs_bb
        energies_bc, xs_bc = self.get_bc(vi)
        avg_bc = self.boltzmann_factor(we, De, vi, T) * self.interpolate(
            energies, energies_bc, xs_bc
        )
        avg = avg_bb + avg_bc

        for vi in range(1, vmax):  # ignore last initial vib state
            energies_bb, xs_bb = self.get_bb(vi)
            xs_bb = self.interpolate(energies, energies_bb, xs_bb)
            avg_bb += self.boltzmann_factor(we, De, vi, T) * xs_bb
            energies_bc, xs_bc = self.get_bc(vi)
            xs_bc = self.interpolate(energies, energies_bc, xs_bc)
            avg_bc += self.boltzmann_factor(we, De, vi, T) * xs_bc
            avg += self.boltzmann_factor(we, De, vi, T) * (xs_bb + xs_bc)

        label = str(T) + r"$\,$K R-matrix"
        if self.summed:
            ax.plot(energies, norm * avg, color=color)
        else:
            ax.plot(
                energies,
                norm * avg_bb,
                color=color,
                linestyle="--",
                label=r"b-b " + label,
            )
            ax.plot(energies, norm * avg_bc, color=color, label=r"b-d " + label)


class ICECProcessor:
    """Plots ICEC cross sections from files calculated using the analytical approach.
    Includes data from R-matrix results.
    - process : identifier for the transition / process to get correct data
    - overlap : if the process includes overlap (i.e. electron transfer contribution)
    - box_length : box length to discretize the continuum vibrational states [Angstrom]
    """

    def __init__(
        self,
        process: str,
        overlap: bool = False,
        box_length: float = 10,
    ):
        self.process = process
        self.overlap = overlap
        self.box_length = box_length

    def read_constR_results(self):
        file_path = DIRECTORY + "results/" + self.process + ".xs.constR.txt"
        self.results_constR = np.loadtxt(file_path, comments="#")

    def read_xs_results(self):
        file_path = DIRECTORY + "results/" + self.process + ".bb.txt"
        self.results_bb = np.loadtxt(file_path, comments="#") # bound-bound transitions
        file_path = DIRECTORY + "results/" + self.process + ".bc." + str(box_length) + "A.txt" 
        self.results_bc = np.loadtxt(file_path, comments="#") # bound-dissociative transitions

    def read_temperature_results(self):
        # temperature dependent cross sections
        file_path = DIRECTORY + "results/" + self.process + ".xs.temperature.txt"
        self.results_temperature = np.loadtxt(file_path, comments="#")

    def set_vmax(self, mu, De, alpha):
        # max v for the morse potential
        lam = np.sqrt(2 * mu * De) / alpha
        self.vi_max = int(lam - 0.5)

    def plot_fixedR(
        self,
        ax: plt.Axes,
        color="tab:red",
        linestyle=":",
        label=False,
        overlap=False,
        summed=False,
    ):        
        """Plots the ICEC cross section at constant R."""
        if not hasattr(self, "results_constR"):
            self.read_constR_results()
        if self.overlap and summed:
            xs = self.results_constR[:, 2] + self.results_constR[:, 1]
        elif overlap:
            xs = self.results_constR[:, 2]
        else:
            xs = self.results_constR[:, 1]
        ax.plot(
            self.results_constR[:, 0],
            xs,
            color=color,
            linestyle=linestyle,
            label=r"$R_\text{e}$" if label else None,
            lw=2
        )

    def get_xs(self, vi, overlap=False, summed=True):
        if not hasattr(self, "results_bb"):
            self.read_xs_results()
        energies = self.results_bb[:, 0]
        if overlap: # electron transfer contribution
            index = 2 * vi + 2
        elif self.overlap:
            index = 2 * vi + 1 # energy transfer contribution for a transition where electron transfer is possible
        else:
            index = vi + 1 # energy transfer contribution
        if self.overlap and summed:
            xs_bb = self.results_bb[:, index - 1] + self.results_bb[:, index]
            xs_bc = self.results_bc[:, index - 1] + self.results_bc[:, index]
        else:
            xs_bb = self.results_bb[:, index]
            xs_bc = self.results_bc[:, index]
        return energies, xs_bb, xs_bc

    def configure_plot(self, ax: plt.Axes):
        ax.tick_params(right=True)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\epsilon$ [eV]")
        ax.set_ylabel(r"$\sigma$ [Mb]")

    def plot_xs(self, ax: plt.Axes, vi, color="tab:red", **kwargs):
        """Plots the ICEC cross section corresponding to bound-bound and 
        bound-dissociative (or continuum) transitions of the dimer."""
        self.configure_plot(ax)
        self.plot_fixedR(ax, color, label=False, overlap=self.overlap, summed=True)
        energies, xs_bb, xs_bc = self.get_xs(vi, self.overlap)
        mask = energies > 0
        ax.plot(energies[mask], xs_bb[mask], color=color, dashes=(5, 3), **kwargs)
        ax.plot(energies[mask], xs_bc[mask], color=color, label=r"model", **kwargs)

    def plot_all_vi(self, ax: plt.Axes, vmax):
        """Plots the individual cross section for every initial vibrational state."""
        self.configure_plot(ax)
        reds = plt.get_cmap("OrRd_r")
        for vi in range(vmax):  # ignore highest initial vibrational state
            energies, xs_bb, xs_bc = self.get_xs(vi, self.overlap)
            if vi == int(vmax / 2):
                label_bb, label_bc = "b-b", "b-d"
            else:
                label_bb, label_bc = "_nolegend_", "_nolegend_"
            rgba = reds(vi / (vmax + 0.5))
            ax.plot(energies, xs_bb, color=rgba, linestyle="--", label=label_bb)
            ax.plot(energies, xs_bc, color=rgba, label=label_bc)

    def plot_terms(self, ax: plt.Axes, vi):
        """Plots the electron transfer and energy transfer contributions to ICEC."""
        self.configure_plot(ax)

        color = "tab:red"
        self.plot_fixedR(ax, overlap=True)
        energies, xs_bb, xs_bc = self.get_xs(vi, overlap=True, summed=False)
        ax.plot(energies, xs_bb + xs_bc, color=color, label="electron tf")

        color = "tab:orange"
        self.plot_fixedR(ax, color=color, overlap=False)
        energies, xs_bb, xs_bc = self.get_xs(vi, overlap=False, summed=False)
        ax.plot(energies, xs_bb + xs_bc, color=color, label="energy tf")

    def plot_xs_temperature(self, ax: plt.Axes, T):
        """Plots the temperature dependent ICEC cross sections."""
        self.configure_plot(ax)
        reds = plt.get_cmap("Reds_r")

        if not hasattr(self, "results_temperature"):
            self.read_temperature_results()
        energies = self.results_temperature[:, 0]

        mask = energies > 0
        for i in range(len(T)):
            t = T[i]
            label = str(t) + r"$\,$K"
            red = reds(i / (len(T) + 1 / len(T)))
            xs = self.results_temperature[:, i + 1]
            ax.plot(energies[mask], xs[mask], color=red, label=label)

    def plot_PR_xs(self, ax: plt.Axes, IP, PI_xs_A, deg_factor, **kwargs):
        """Plots the Photorecombination Cross section [Mb]."""
        PR_xs = np.array([])
        energies = np.geomspace(0.01 * EV2HARTREE, 10 * EV2HARTREE, 200)
        for electronE in energies:
            hbarOmega = electronE + IP
            PI_xs = PI_xs_A(hbarOmega * HARTREE2EV) * MB2AU
            xs = deg_factor * hbarOmega**2 / (2 * electronE * c**2) * PI_xs
            PR_xs = np.append(PR_xs, [xs * AU2MB])
        mask = PR_xs > 0
        ax.plot(energies[mask] * HARTREE2EV, PR_xs[mask], **kwargs)


# ===== System specific functions =====
def define_title_of_legend_4(process):
    if process == "2A1.1B1":
        return r"(a) B to A"
    elif process == "1B1.2A1":
        return r"(b) A to B"
    elif process == "2A1.1A1":
        return r"(c) B to X"
    elif process == "1A1.2A1":
        return r"(d) X to B"


def define_title_of_legend_2(process):
    if process == "2A1.1A1":
        return r"(a) B to X"
    elif process == "1A1.2A1":
        return r"(b) X to B"


def set_legend(ax, process, num_plots=4):
    if process == "2A1.1B1" or process == "2A1.1A1":
        loc = "upper right"
    else:
        loc = "lower right"
    if num_plots == 2:
        title = define_title_of_legend_2(process)
    elif num_plots == 4:
        title = define_title_of_legend_4(process)
    else:
        print("invalid num_plots")
    legend = ax.legend(loc=loc, title=title)
    title = legend.get_title()
    title.set_weight("bold")


def get_values_for_initial_state(process):
    """Takes input file and attributes values to corresponding transitions."""
    if process == "2A1.1B1" or process == "2A1.1A1":
        IP_A = IP_He * EV2HARTREE
        PI_xs_A = PI_xs_He
        deg_factor = degeneracyHe
        mu, we, req, De = stateB
    else:
        IP_A = IP_Ne * EV2HARTREE
        PI_xs_A = PI_xs_Ne
        deg_factor = degeneracyNe
    if process == "1A1.2A1":
        mu, we, req, De = stateX
    elif process == "1B1.2A1":
        mu, we, req, De = stateA
    we *= WAVENUMBER2HARTREE
    De *= WAVENUMBER2HARTREE
    alpha = we * np.sqrt(mu / 2 / De)
    lam = np.sqrt(2 * mu * De) / alpha
    vmax = int(lam - 0.5)
    return IP_A, PI_xs_A, deg_factor, we, De, vmax


def set_rmatrix_shift(process):
    if process in ["2A1.1A1", "2A1.1B1"]:
        return -2.94  # eV
    else:
        return 0


# ===== Plot generators =====
def generate_xs_plots(plot_setup, process, overlap=False, box_length=10):
    """Generates figures displaying ICEC cross section corresponding to bound-bound 
    and bound-dissociative (or continuum) transitions of the dimer."""
    width, height, xmin, xmax, ymin, ymax = plot_setup
    fname = (
        DIRECTORY + "plots/" + process + ".xs." + str(box_length) + "A.pdf"
    )
    os.makedirs('./plots', exist_ok=True)
    IP_A, PI_xs_A, deg_factor, we, De, vmax = get_values_for_initial_state(process)
    with PdfPages(fname) as pdf:
        for vi in range(vmax + 1):
            fig, ax = plt.subplots(figsize=(width, height))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            icec = ICECProcessor(process, overlap)
            Rmatrix = RMatrixProcessor(process)
            icec.plot_PR_xs(
                ax,
                IP_A,
                PI_xs_A,
                deg_factor,
                color="grey",
                linestyle="--",
                label=r"$\sigma_\text{PR}$",
                lw=2
            )
            Rmatrix.plot_constR(ax, set_rmatrix_shift(process), lw=2)
            Rmatrix.plot_xs(ax, vi, lw=2)
            icec.plot_xs(ax, vi, lw=2)

            set_legend(ax, process, num_plots=4)
            plt.tight_layout()
            pdf.savefig(fig)  # , bbox_inches = "tight"
            plt.close(fig)


def generate_all_vi_plots(plot_setup, process, overlap=False, box_length=10):
    """Generates figure displaying ICEC cross section for every initial vibrational state individually."""
    width, height, xmin, xmax, ymin, ymax = plot_setup
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    IP_A, PI_xs_A, deg_factor, we, De, vmax = get_values_for_initial_state(process)
    icec = ICECProcessor(process, overlap)
    icec.plot_fixedR(ax, label=True, overlap=overlap, summed=True)
    icec.plot_all_vi(ax, vmax)

    set_legend(ax, process, num_plots=4)
    plt.tight_layout()
    fname = (
        DIRECTORY + "plots/" + process + ".all-vi." + str(box_length) + "A.pdf"
    )
    os.makedirs('./plots', exist_ok=True)
    fig.savefig(fname)


def generate_term_plots(plot_setup, process, overlap=True, box_length=10):
    """Generates figures displaying electron transfer and energy transfer contributions to ICEC."""
    width, height, xmin, xmax, ymin, ymax = plot_setup
    fname = (
        DIRECTORY + "plots/" + process + ".en-el-transfer." + str(box_length) + "A.pdf"
    )
    os.makedirs('./plots', exist_ok=True)
    IP_A, PI_xs_A, deg_factor, we, De, vmax = get_values_for_initial_state(process)
    with PdfPages(fname) as pdf:
        for vi in range(vmax + 1):
            fig, ax = plt.subplots(figsize=(width, height))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            Rmatrix = RMatrixProcessor(process, summed=True)
            Rmatrix.plot_constR(ax, set_rmatrix_shift(process))
            Rmatrix.plot_xs(ax, vi)
            icec = ICECProcessor(process, overlap)
            icec.plot_terms(ax, vi)

            set_legend(ax, process, num_plots=2)
            plt.tight_layout()
            pdf.savefig(fig)  # , bbox_inches = "tight"
            plt.close(fig)


def generate_temperature_plot(plot_setup, process, T, overlap=False, box_length=10):
    """Generates figure showing temperature dependent ICEC cross sections."""
    width, height, xmin, xmax, ymin, ymax = plot_setup
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    IP_A, PI_xs_A, deg_factor, we, De, vmax = get_values_for_initial_state(process)
    blues = plt.get_cmap("Blues_r")
    Rmatrix = RMatrixProcessor(process=process, summed=True)
    for t in T:
        blue = blues(T.index(t) / (len(T) + 1 / len(T)))
        Rmatrix.plot_xs_temperature(ax, t, we, De, vmax, color=blue)

    icec = ICECProcessor(process, overlap)
    icec.plot_xs_temperature(ax, T)

    set_legend(ax, process, num_plots=4)
    plt.tight_layout()

    fname = (
        DIRECTORY + "plots/" + process + ".temperature." + str(box_length) + "A.pdf"
    )
    os.makedirs('./plots', exist_ok=True)
    fig.savefig(fname)


# ========================================
width, height = 8, 5  # of the figure
box_length = 10  # angstrom, equivalent to the box_length defined in the header of the cross section files
lmax = 10
T = [15, 77, 298]  # K, must be equivalent to T in the header of the temperature dependent file

# ========================================
process = "2A1.1A1"
overlap = True
xmin, xmax = 0, 5  # eV, plot limits
ymin, ymax = 1e-4, 1e6

plot_setup = width, height, xmin, xmax, ymin, ymax
generate_xs_plots(plot_setup, process, overlap, box_length)
plot_setup = width, height, xmin, xmax, 1e-1, 1e5
#generate_term_plots(plot_setup, process, overlap, box_length)
plot_setup = width, height, xmin, xmax, 1, 1e5
#generate_all_vi_plots(plot_setup, process, overlap, box_length)
plot_setup = width, height, xmin, xmax, 10, 1e5
#generate_temperature_plot(plot_setup, process, T, overlap, box_length)

# ===========================
process = "2A1.1B1"
xmin, xmax = 0, 5  # eV
ymin, ymax = 1e-4, 1e2

plot_setup = width, height, xmin, xmax, ymin, ymax
generate_xs_plots(plot_setup, process)
plot_setup = width, height, xmin, xmax, 1e-3, 1e2
#generate_all_vi_plots(plot_setup, process)
plot_setup = width, height, xmin, xmax, 1e-1, ymax
#generate_temperature_plot(plot_setup, process, T)

# ========================================
process = "1B1.2A1"
xmin, xmax = 2.5, 8  # eV
ymin, ymax = 1e-4, 1  # eV

plot_setup = width, height, xmin, xmax, ymin, ymax
generate_xs_plots(plot_setup, process)
plot_setup = width, height, xmin, xmax, 1e-2, 1
#generate_all_vi_plots(plot_setup, process)
plot_setup = width, height, xmin, xmax, 1e-2, 1
#generate_temperature_plot(plot_setup, process, T)

# ===========================
process = "1A1.2A1"
overlap = True
xmin, xmax = 2.5, 8  # eV
ymin, ymax = 1e-9, 1e5  # eV

plot_setup = width, height, xmin, xmax, ymin, ymax
generate_xs_plots(plot_setup, process, overlap)
#generate_all_vi_plots(plot_setup, process, overlap)
plot_setup = width, height, xmin, xmax, 1e-6, 1e5
#generate_term_plots(plot_setup, process, overlap)
plot_setup = width, height, xmin, xmax, 1e-6, 1e4
#generate_temperature_plot(plot_setup, process, T, overlap)