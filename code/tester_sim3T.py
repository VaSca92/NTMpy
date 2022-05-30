from Sim3T import Sim3T
import Visual as vs
from Source import source
import numpy as np
import matplotlib.pyplot as plt

#Define the source, responsible for heating
s = source()
s.type_x  = "TMM"
s.type_t  = "Gaussian"
s.setLaser( 50, 1e-13) 
s.delay = 5e-13
s.polarization  = "p"
s.angle = np.pi/4


length      = 20e-9
density     = 8.908e+3 
n_index     = 1.7163
C_l         = 2.2e6/density
gamma       = 6e3/density
#The units of C_tot = J/kgK --> don´t devide over density any more!
C_tot       = lambda T: np.piecewise(T, [T<600, (T>=600) & (T<700),T>= 700 ], \
                 [lambda T:1/0.058* (13.69160 + 82.49509*(T/1000) - 174.955*(T/1000)**2 + 161.6011*(T/1000)**3),
                  lambda T:1/0.058* (1248.045 - 1257.510*(T/1000) - 165.1266/(T/1000)**2),
                  lambda T:1/0.058* (16.49839 + 18.74913*(T/1000) - 6.639841*(T/1000)**2 + 1.717278*(T/1000)**3)])
C_e         = lambda T: gamma *T
#Extract the slope of the total heat capacity before and after curie temperature
temp = np.linspace(300,2000,5000)
indexl = temp <= 500
indexh = temp > 750
z1  = np.polyfit(temp[indexl],C_tot(temp[indexl]),1)
Ce1 = np.poly1d(z1)
coef1 = Ce1.coef
print("Linear approx before Curie temp:")
print(Ce1)
z2  = np.polyfit(temp[indexh],C_tot(temp[indexh]),1)
Ce2 = np.poly1d(z2)
coef2 = Ce2.coef
print("Linear approx after Curie temp:")
print(Ce2)
gammaS = coef1[0]-coef2[0]
print(f"Difference of slopes gives gammaS: {gammaS:.3f}")
C_s = lambda Ts: gammaS * Ts
#Conductivity is not so important as we assume uniform heating all get the same conductivity
k = 1
#Coupling constants taken from paper
G_el = 8e17
G_se = 6e17
G_ls = 0.3e17 


sim = Sim3T()
sim.time_step = 6e-15
s.refraction = [n_index]
s.absorption = [20e-9]
sim.setSource(s)
sim.addLayer(length,[k,k,k],[C_e,C_l,C_s],density,[G_el,G_se,G_ls], 20)
sim.final_time = 6e-12
#To get the raw output in form of arrays
[phi, x] = sim.run()


t = np.arange(sim.start_time, sim.final_time, sim.time_step)
vs.average(x,t,phi)