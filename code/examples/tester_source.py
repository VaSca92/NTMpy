import sys
sys.path.insert(0, '../')

from Source import source
import numpy as np
import matplotlib.pyplot as plt


# Initialize source
s = source() # default option, Gaussian pulse
s.setLaser(1, 1e-12)
s.delay       = 2e-12       # time the maximum intensity hits

s.angle = np.pi/4

s.refraction  = [1+.1j]
s.wavelength  = 400e-9
s.thickness   = [10000e-9]

end = np.sum(s.thickness)
x = np.linspace( 0,   end, 4000)
t = np.linspace( 0, 4e-12, 2000)

SL = s.matrix(x,t)
s.type_x = 'tmm'

s.polarization = 's'
STs = s.matrix(x,t)

s.polarization = 'p'
STp = s.matrix(x,t)

#plt.plot(x,SL[10,:],x,ST[10,:])
#plt.plot(x,STp[10,:],x,STs[10,:])

plt.plot(t,np.cumsum(np.sum(STs,1))*t[1]*x[1])
print(np.sum(STs*t[1]*x[1]))
plt.plot(t,np.cumsum(np.sum(STp,1))*t[1]*x[1])
print(np.sum(STp*t[1]*x[1]))      
      