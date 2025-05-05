# Define Constants and parameters for the (NeHe)^+ dimer

# energy
EV2HARTREE = 3.67493e-2
HARTREE2EV = 27.2114
WAVENUMBER2HARTREE = 4.55633e-6 # cm^-1

# length
M2BOHR = 18897259885.789
PM2BOHR = 1e-12 * M2BOHR 
ANGSTROM2BOHR = 1e-10* M2BOHR

# ===== Atomic Properties =====
# from https://doi.org/10.1016/0092-640X(76)90015-2
IP_He = 198310.6691*WAVENUMBER2HARTREE*HARTREE2EV
IP_Ne = 173929.75*WAVENUMBER2HARTREE*HARTREE2EV

def PI_xs_Ne(E):
    if E > 21.56 and E < 42.03:
        return -3.63076977e-5*E**4 + 5.41943012e-03*E**3 - 3.07876745e-1*E**2 + 7.83793553*E - 6.61201651e1
    else:
        return 0

def PI_xs_He(E):
    if E > 24.60 and E < 45.08:
        return -3.39517252e-4*E**3 + 4.48307894e-2*E**2 - 2.10228697*E + 3.71807537e1
    else:
        return 0

degeneracyNe = 1./6
inputNeHe = (degeneracyNe, IP_Ne, IP_He, PI_xs_Ne, PI_xs_He)

degeneracyHe = 0.5
inputHeNe = (degeneracyHe, IP_He, IP_Ne, PI_xs_He, PI_xs_Ne)

# ===== Morse Potential =====
# from https://doi.org/10.1016/S0009-2614(99)01179-3
# state = (mu [proton mass], we [cm-1], req [Angstrom], De [cm-1])
mu = 6090       # proton mass
Req_X = 1.43    # Angstrom
Req_A = 2.42    # Angstrom
Req_B = 2.66    # Angstrom
stateX = (mu, 911, 1.43, 5200) # 1A1 = X
stateA = (mu, 152, 2.42, 283)  # 1B1, 1B2 = A
stateB = (mu, 152, 2.66, 343)  # 2A1 = B

# ===== Parameters for electron transfer =======
alpha = 1.63

# r from https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.200800987
r_He = 46 * PM2BOHR
a_He = alpha * r_He
r_Ne = 67 * PM2BOHR
a_Ne = alpha * r_Ne

Cbar = 9
d = 1.9 * EV2HARTREE

overlapNeHe = (a_Ne, a_He, Cbar, d)
overlapHeNe = (a_He, a_Ne, Cbar, d)
