import sys

sys.path.insert(0, '../Classes')

from Source import source
import numpy as np
import matplotlib.pyplot as plt

# Initialize source
s = source()  # default option, Gaussian pulse
s.setLaser(1, 1e-12)
s.delay = 2e-12  # time the maximum intensity hits

s.angle = np.pi / 4 * 0

s.refraction = [1 + 4j, 2j]
s.wavelength = 400e-9
s.thickness = [10000e-9, 2e-8]

s.refraction = [1.0433 + 3.0855j, 1.0454 + 3.2169j, 2.0150 + 2.8488j, 1.766]
s.thickness = [3e-9, 15e-9, 5e-9, 25e-9]

s.refraction = [1 + .1j, 1 + .1j]
s.thickness = [4e-6, 4e-6]
s.absorption = [1e-7, 1e-7]

end = np.sum(s.thickness)
x = np.linspace(0, end, 4000)
t = np.linspace(0, 4e-12, 2000)

SL = s.matrix(x, t)
s.type_x = 'tmm'

s.polarization = 's'
STs = s.matrix(x, t)

s.polarization = 'p'
STp = s.matrix(x, t)

plt.plot(x, SL[:, 0])
# plt.plot(x,SL[10,:],x,ST[10,:])
# plt.plot(x,STp[10,:],x,STs[10,:])

# plt.plot(t,np.cumsum(np.sum(SL,0))*t[1]*x[1])
print(np.sum(SL * t[1] * x[1]))
# plt.plot(t,np.cumsum(np.sum(STs,0))*t[1]*x[1])
print(np.sum(STs * t[1] * x[1]))
# plt.plot(t,np.cumsum(np.sum(STp,0))*t[1]*x[1])
print(np.sum(STp * t[1] * x[1]))
