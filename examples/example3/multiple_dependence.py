import sys
sys.path.insert(0, '../../')

from Sim2T import Sim2T
import Visual as vs
from Source import source
import numpy as np

# Case 3 ======================================================================

sim = Sim2T()
# Add Nichel
sim.addLayer( 50e-9, [lambda Te, Tl: 91 * Te / Tl, lambda T: 1065*T, 1, 3.6e17)
# Add Gold
sim.addLayer(100e-9, [lambda Te, Tl: 91 * Te / Tl], lambda T:   71*T, 1, 2.1e16)


src = source()
src.absorption = 

sim.setSource(src)
# =============================================================================
