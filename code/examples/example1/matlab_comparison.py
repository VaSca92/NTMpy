import sys
sys.path.insert(0, '../../')

from Sim2T import Sim2T
import Visual as vs
from Source import source
import numpy as np

# Case 1 ==================================================================
# Create Source
s = source()
s.type_x = "lb"
s.absorption = [45e-9, 45e-9]
s.peak  = 2e+18*45e-9
s.time  = 2e-12
s.delay = 2e-12
# Instantiate the Sim2T class
sim = Sim2T()
# Add layers (Length,conductivity,heatCapacity,rho,coupling)
sim.addLayer(40e-9, [ 6,  1], [lambda T: .112*T, 450], 6500, 5e17, 6)
sim.addLayer(80e-9, [12,  1], [lambda T: .025*T, 730], 5100, 5e17, 10)

sim.setSource(s)

sim.final_time = 12e-12

[x, t, phi] = sim.run()

#vs.compare(x,t,phi[0],phi[1], 2)

vs.spaceVStime(x, t, phi[0])
vs.average(x,t,phi)

    
np.savetxt('x.txt', x)
np.savetxt('t.txt', t)
np.savetxt('phie.txt', phi[0])
np.savetxt('phil.txt', phi[1])

# -------------------------------------------------------------------------

