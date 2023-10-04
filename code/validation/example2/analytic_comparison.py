import sys
sys.path.insert(0, '../../')

from Sim2T import Sim2T
import Visual as vs
from Source import source
import numpy as np

# Case 2 ==================================================================

L  = 5e-6      # Length of the Grating
Ce = 2e+4      # Specific Heat Electrons
Cl = 2.5e6     # Specific Heat Lattice
ke = 3.2e+1    # Conductivity Electrons
kl = 2.75      # Conductivity Lattica
G  = 3e+16     #Exchange constant
# Define the Object
sim = Sim2T()
# Define Media
sim.addLayer(L, [ke,kl], [Ce, Cl], 1, G)
# Define Initial Condition
sim.init_E = lambda x: np.sin(2*np.pi*x/L)
sim.init_L = 0
# Assign Boundary Condition type
sim.LBCT_E = 0
sim.RBCT_E = 0
sim.LBCT_L = 0
sim.RBCT_L = 0
sim.LBCV_E = 0
sim.RBCV_E = 0
sim.LBCV_L = 0
sim.RBCV_L = 0
# Chose Final  Time
sim.final_time = 8000

# RUN ...
[x, t, phi] = sim.run()
   

q = 2*np.pi/L
# Find Analitical Values of gamma1 and gamma2
A = 1; B = (ke/Ce+kl/Cl)*q**2 + G*(1/Ce+1/Cl)
C = G*(ke+kl)*q**2/(Ce*Cl) + q**4*(ke*kl)/(Ce*Cl)
# Resolutive formula for 2nd grade equations
delta = B**2-4*A*C

index = np.where(phi[0][:,0] == np.max(phi[0][:,0]))[0]

dt = sim.time_step
g2 = -(phi[0][index, -2]-phi[0][index, -1])/(dt*phi[0][index, -1])
g3 = -np.log(phi[0][index, -2]/phi[0][index, -1])/dt
g1 = - (-B+np.sqrt(delta))/2


print("Analytic value of the time constant:    " + str(1/g1))
print("Numeric (1) value of the time constant: " + str(-1/g2[0]))
print("Numeric (2) value of the time constant  " + str(-1/g3[0]))

# -------------------------------------------------------------------------
