import numpy as np
import matplotlib.pyplot as plt


kB = 1.3806e-23
h = 1.05457e-34
me = 9.1094e-31
me = 1.0*me
qe = 1.6022e-19
ep = 8.8542e-12
eV = 1.60217e-19


h = 1.05410e-27   # erg*s
me = 9.10910e-28  # g
me = me * 1.0
qe = 4.810e-10    # statC
kB = 1.3810e-16   # erg/K
ep = 1            # statC
eV = 1.60217e-12


EF = 5.5 * eV 
#EF = 11.64 * eV

#vF = np.sqrt(2*EF/me)
kF = np.sqrt(2*EF*me) / h


#mu = np.load('chemical_mu.npy')*qe
#Tm = np.load('chemical_Te.npy')*kB

mu_Au = np.loadtxt('../data/mu_Au.dat')
mu = mu_Au[:,1] * eV + EF
Tm = mu_Au[:,0]*kB*1e4


Te = np.linspace(300, 9000, 20)*kB
#Te = np.array([300])*kB
kee = np.zeros(Te.shape)
vint = np.zeros(Te.shape)



K1 = np.linspace(0.75*kF, 1.25*kF, 256)
#K1 = np.linspace(0.25*kF, 8.25*kF, 256)
K1 = K1[1:-1]
q, k2 = np.meshgrid( np.linspace(0, 8*kF, 256), np.linspace(0, 1.75*kF, 128))

s2 = np.zeros(q.shape)
e2 = np.zeros(q.shape)

vee = np.zeros(K1.shape)
Ia  = np.zeros(K1.shape)
Ib  = np.zeros(K1.shape)
Id  = np.zeros(K1.shape)


#U = ( 4*np.pi*qe**2 / (q**2+(1.41*kF)**2) )**2 * k2
U = ( 4*np.pi*qe**2 / (q**2+(1.4*kF)**2) / ep )**2 * k2


def regularize(E1,E2,EA,EB):
    
    S = np.zeros(E2.shape)
       
    x, y = np.where( EA + E2 > 37 )
    S[x,y] += EA[x,y] + E2[x,y]
    x, y = np.where( EA + E2 < 37 )
    S[x,y] += np.log( 1 + np.exp(EA[x,y] + E2[x,y]) )

    x, y = np.where( EA - E1 > 37 )
    S[x,y] -= EA[x,y] - E1
    x, y = np.where( EA - E1 < 37 )
    S[x,y] -= np.log( 1 + np.exp(EA[x,y] - E1) )
    
    x, y = np.where( EB - E1 > 37 )
    S[x,y] += EB[x,y] - E1
    x, y = np.where( EB - E1 < 37 )
    S[x,y] += np.log( 1 + np.exp(EB[x,y] - E1) )
    
    x, y = np.where( EB + E2 > 37 )
    S[x,y] -= EB[x,y] + E2[x,y]
    x, y = np.where( EB + E2 < 37 )
    S[x,y] -= np.log( 1 + np.exp(EB[x,y] + E2[x,y]) )
    
    return S 

def dfe(e, T):
    f = np.exp(-e/T)/T
    x = np.where(e/T < 18)
    f[x] = np.exp(e[x]/T)/(1+np.exp(e[x]/T))**2/T
    return f



for j, T in enumerate(Te):
    
    m  = np.interp(T,Tm,mu)
    dm = ( np.interp(T*(1+1e-5),Tm,mu) - np.interp(T*(1-1e-5),Tm,mu) ) / (T*2e-5)
    E2 = ( k2**2 * h**2/(2*me) - m )/T
    x, y = np.where(E2 < 37)

    
    for i, k1 in enumerate(K1):
    
        E1 = ( k1**2 * h**2 /(2*me) - m ) / T
    
        a1 = ( -(2*k1*q + q**2) * h**2 / (2*me)  ) / T
        a2 = ( +(2*k1*q - q**2) * h**2 / (2*me)  ) / T
        b1 = ( +(q**2 - 2*k2*q) * h**2 / (2*me)  ) / T
        b2 = ( +(q**2 + 2*k2*q) * h**2 / (2*me)  ) / T   

        if E1 < 37:
            s1 = (1 + np.exp(E1)) / (1 - np.exp(E1+E2))
        else:
                s1 = - np.exp(-E2)
        
        
        s1[x,y] *= np.exp(E2[x,y]) / (1 + np.exp(E2[x,y]))


        s2 = regularize(E1, E2, b1, a2)
        Sa = T * s1 * s2
        Ma = ( (np.abs(k2-k1)< q) * (q<k1) ) + ( (np.abs(k2-q)<k1) * (q>k1) )
        Ia[i] = np.trapz(np.trapz( Ma * U * Sa, k2, axis=0), q[0], axis = 0)
    
    
        s2 = regularize(E1, E2, a1, a2)
        Sb = T * s1 * s2
        Mb  = k2 > q + k1
        Ib[i]  = np.trapz(np.trapz( Mb  * U * Sb, k2, axis=0), q[0], axis = 0)


        s2 = regularize(E1, E2, b1, b2)
        Sd = T * s1 * s2
        Md  = (q < k1) * (k2 < - q + k1)
        Id[i]  = np.trapz(np.trapz( Md  * U * Sd, k2, axis=0), q[0], axis = 0)

        #plt.figure()
        #plt.pcolor(q/kF, k2/kF, Sa*Ma*U + Sb*Mb*U + Sd*Md*U )

    print(j)

    vee = (Ia + Ib + Id ) / ( 16*np.pi**3 * h * K1 * (h**2/(2*me))**2 )

    e = K1**2*h**2/(2*me)
    v2 = K1**2*h**2/me**2
    diff = (dm + (e-m)/T)

    kee[j] = kB/3/np.pi**2 * np.trapz( (e-m) * dfe(e-m,T) * diff * v2 * K1**2 / vee, K1)
    kee[j] *= 1e-5

 

    vint[j] = np.trapz( dfe(e-m,T) * vee, e) * (1 + np.exp(-m/T))


C_Au = np.loadtxt('../data/Ce_Au.dat')
C_Au[:,0] *= 1e4; C_Au[:,1] *= 1e5
Ce = lambda T: np.interp(T, C_Au[:,0], C_Au[:,1])

veff = Ce(Te/kB)*1.5e6**2/(3*kee)

kef = 1.5e6**2*Ce(Te/kB)/(veff + 1.2e11*300)/3

'''
plt.figure()
plt.plot(Te/kB, kee)

plt.figure()
plt.plot(Te/kB, vef)
'''

'''
plt.figure()
plt.plot(K1/kF,vee)
plt.grid()
'''  
'''
plt.figure()
plt.pcolor(q/kF, k2/kF, np.abs(Sa*Ma*U + Sb*Mb*U + Sd*Md*U) )
'''
    

'''
plt.figure()
plt.plot(K1/kF, (e-m) * dfe(e-m,T) * kvsk * v2 * K1**2 / vee)
plt.grid()
'''

