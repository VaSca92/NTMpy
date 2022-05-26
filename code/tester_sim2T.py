from Sim2T import simulation
from Visual import *
from Source import source

# Case 1 ======================================================================

s = source()
#s.type_x = "beerlambert"
s.absorption = [1e-8, 1e-8]
s.refraction = [1, 1]
s.time = 2e-12
s.delay = 2e-12

sim = simulation()
#add layers (Length,conductivity,heatCapacity,rho,coupling)
sim.addLayer(40e-9, [ 6,  1], [lambda T: .112*T, 450], 6500, 5e17, 12)
sim.addLayer(80e-9, [12,  1], [lambda T: .025*T, 730], 5100, 5e17, 12)

sim.setSource(s)

sim.final_time = 10e-12

#[phi_E,phi_L,x] = sim.run()
[phi,x] = sim.run()
#print(sim)

t = np.arange(sim.start_time, sim.final_time, sim.time_step)
#compare(x,t,phi[0],phi[1], 2)

spaceVStime(x, t, phi[0])
#surface(x,t,s.matrix(x,t,sim.length))
#print(phi[0])
# -----------------------------------------------------------------------------

# Case 2 ======================================================================
'''
L  = 5e-6      # Length of the Grating
Ce = 2e+4      # Specific Heat Electrons
Cl = 2.5e6     # Specific Heat Lattice
ke = 3.2e+1    # Conductivity Electrons
kl = 2.75      # Conductivity Lattica
G  = 3e+16     #Exchange constant
# Define the Object
sim = simulation()
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
[phi, x] = sim.run()
t = np.arange(sim.start_time, sim.final_time, sim.time_step)

#vs.compare(x,t,phi[0],phi[1],1)
#plt.plot(x, phi[0][1,:], x, phi[1][1,:], x, phi[0][-1,:], x, phi[1][-1,:])
#plt.grid();plt.show()

q = 2*np.pi/L
# Find Analitical Values of gamma1 and gamma2
A = 1; B = (ke/Ce+kl/Cl)*q**2 + G*(1/Ce+1/Cl)
C = G*(ke+kl)*q**2/(Ce*Cl) + q**4*(ke*kl)/(Ce*Cl)
# Resolutive formula for 2nd grade equations
delta = B**2-4*A*C

index = np.where(phi[0][0,:] == np.max(phi[0][0,:]))[0]

dt = sim.time_step
g2 = -(phi[0][-2, index]-phi[0][-1, index])/(dt*phi[0][-1, index])
g3 = -np.log(phi[0][-2, index]/phi[0][-1, index])/dt
g1 = - (-B+np.sqrt(delta))/2


print("Analytic value of the time constant:    " + str(1/g1))
print("Numeric (1) value of the time constant: " + str(-1/g2[0]))
print("Numeric (2) value of the time constant  " + str(-1/g3[0]))
'''
# -----------------------------------------------------------------------------


# Case 3  ---------------------------------------------------------------------
'''
#Compute phi_E (matrix of temperature) and So_map (matrix of source) first
Ce 	  = lambda Te: 0.112*Te
Ce_prm = lambda Te: 0.5*0.112*Te**2
# Sim2T
sim = simulation()
sim.addLayer(1e-9, [1, 0], [Ce, 1], 1, 0)
sim.final_time = 1e-9
s = source()
s.peak = 1
s.absorption = 1e6
s.time = 2e-12
s.delay = 4e-12
#Compute integral over space
Ce_map = (Ce_prm(phi_E) - Ce_prm(300*np.ones_like(phi_E)))*(1/u.ps)
mydx  = x[1]-x[0]
Ce_int = np.trapz(Ce_map,x,axis = 1)
So_int = np.trapz(So_map,x,axis = 1)
#...
plt.figure()
plt.plot(t,Ce_int)
plt.plot(t,So_int)
'''
# -----------------------------------------------------------------------------
