import sys
sys.path.insert(0, './code')

from Sim2T import Sim2T
import Visual as vs
from Source import source
import numpy as np

# Case 1 ==================================================================
# Create Source
sr = source()
sr.type_x = "lb"
sr.absorption = [45e-9,45e-9]
sr.peak  = 2e+13
sr.time  = 2e-12
sr.delay = 2e-12
# Instantiate the Sim2T class
sim = Sim2T()
# Add layers (Length,conductivity,heatCapacity,rho,coupling)
sim.addLayer(40e-9, [ 6,  0], [lambda T: .112*T, 450], 6500, 5e17, 10)
sim.addLayer(40e-9, [12,  0], [lambda T: .025*T, 730], 5100, 5e17, 12)

sim.setSource(sr)

sim.final_time = 12e-12

[x, t, phi] = sim.run()

#vs.compare(x,t,phi[0],phi[1], 2)

vs.spaceVStime(x, t, phi[1])
vs.average(x,t,phi)

    
np.savetxt('space.txt', x)
np.savetxt( 'time.txt', t)
np.savetxt('temp_elec.txt', phi[0])
np.savetxt('temp_latt.txt', phi[1])

# -------------------------------------------------------------------------

