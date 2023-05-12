
# All lengths are in angstroms
DISCARDED_WIDTH_EACH_SIDE = 100.0

HEIGHT = 50.0
WIDTH = 500.0 + DISCARDED_WIDTH_EACH_SIDE * 2

X_BOTTOM_LEFT, Y_BOTTOM_LEFT = (0.0, 0.0)
X_TOP_RIGHT, Y_TOP_RIGHT = (WIDTH, HEIGHT)

# The potentials 
# In volts
INITIAL_TOP_POTENTIAL = 0.0
BOTTOM_POTENTIAL = 0.0

# Initial current
INITAL_CURRENT = 0.0

# Electrical conductivity of oxide
# In sieverts/angstrom
SIGMA_HRS = 3.0 * 1e-13

# Electrical conductivity of defect
# In sieverts/angstrom
SIGMA_LRS = 3.5 * 1e-6

# Pre-exponential factor
# 1 / (seconds * angstrom**3)
G_0 = 1.0

# Area factor
# In angstroms
A_F = 500.0

# Defect formation energy
# In electron volts
E_A = 5.9

# Bond polarization factor
# In electron angstroms
B = 91.8

# Defect radius
# In angstroms
R = 1.4

# Boltzmann constant
# in electron volts / kelvin
K_B = 8.617 * 1e-5

# The temperature
# In kelvin
T = 300.0

# Time simulation step
# In seconds
DELTA_T = 1 * 1e-2

# Initial time
# In seconds
INITIAL_TIME = 0.0

# Compliance current
# In amperes
I_CC = 1 * 1e-2

# Voltage ramp-up rate
# In volts / second
VR = 5.0

# This is some 0-dimensional magic number
INITIAL_MESH_RESOLUTION = 150

# Degree of Lagrangian elements
FS_DEGREE = 3

# Population size for the genetic algorithm
POPSIZE = 8

# Number of polygon points which can be tuned
N_POLY_TUNE = 11

# Number of discrete top electrode protrusion heights
N_HEIGHTS = 11

# Number of simulations run to establish the goodness of a unit
N_SIMS_PER_UNIT = 10