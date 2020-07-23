import numpy as np


'''
=========================== Environment ==================================
initial parameters
At the moment it considered to be free space
'''

c0 = 299792458  #speed of light
eta = 377      #Free space impedance

#------------ permeability---------------
mu0 = 4*np.pi/(10**7)   #magnetic permeability of free space
mu_r = 1    #reletive permeability
mu = mu0 * mu_r

#------------- permittivity -------------
eps0 = 8.8541878128/(10**12)     #Vacuum permittivity
eps_r = 1       #reletive permittivity
eps = eps0 * eps_r

#------------ electron parameters------------
# For a collisionless ionosphere and neglecting geomagnetic field effects
n_e = 0      # electron number density of the media, 0 for free space
q_e = 1.602176634/10**19       # electron charge
m_e = 9.1093837015/10**31      # electron mass









