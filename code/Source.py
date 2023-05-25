# ========================================================================================
#   TITLE: NTMpy source
# ----------------------------------------------------------------------------------------
#        Authors: Valentino Scalera, Lukas Alber
#        Version: 2.0
#   Dependencies: numpy, matplotlib, bspline, tqdm
# ----------------------------------------------------------------------------------------
import numpy as np

# ========================================================================================
# SOURCE CLASS
# ========================================================================================
class source(object):
# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def __init__(self):
        self.type_t       = 'Gaussian'
        self.type_x       = 'LambertBeer'

        self.peak         = 5e10
        self.time         = 2e-12
        self.delay        = 0

        self.polarization = 's'
        self.angle      =  0
        self.refraction = [1]
        self.absorption = [1]

        self.thickness  = [ ]
        
        self.time_step_hint = np.inf


# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def getProperties(self): # to depict the properties of the object
        for i in (self.__dict__):
            print(i,' : ',self.__dict__[i])

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def __repr__(self):
        return('Source')

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def matrix(self, x, t):

        if not isinstance(self.absorption, list):
            self.absorption = [self.absorption]

        xmg,tmg = np.meshgrid(x,t)
        if self.type_t.lower() in ["gaussian","gauss"]:
            fun_t = np.exp(-(tmg-self.delay)**2/(2*self.time**2))
        else:
            print("!!!  Source Error: Unknown time profile  !!!")


        if self.type_x.lower() in ["beerlambert","beer","lambert","lambertbeer","lb"]:
            lam = 1/(self.absorption[0]*np.cos(self.angle))
            fun_x = lam*np.exp(-lam*xmg)
        elif self.type_x.lower() in ["tmm","reflected","reflection"]:
            fun_x = np.tile(self.transfer_matrix(x), [len(t), 1])
        else:
            print("!!!  Source Error: Unknown space profile !!!")


        S_matrix = self.peak * fun_x * fun_t
        #Clear for boundary conditions in Simulation core
        S_matrix[:,0] = 0; S_matrix[:,-1] = 0

        self.stored = S_matrix
        return S_matrix

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def transfer_matrix(self, x):

        layer_num = len(self.thickness)
        if len(self.refraction) != layer_num:
            print("!!! Source Error: Unconsistent number of layer !!!")
            print("Number of layers:  " + str(len(layer_num)))
            print("Number of refraction indices:  " + str(len(self.refraction)))
        
        self.wave = []
        self.k  = []
        self.Tn = []
        self.Rn = []
        self.Mn = []
        self.M  = np.eye(2)
        
        # Air wavevector
        k0  = np.array([np.sin(self.angle), np.cos(self.angle)]) 
        k0 *= 2 * np.pi / self.wavelength
        self.k.append(k0)
        
        refraction = np.hstack([1, self.refraction, 1])
        thickness  = np.hstack([0, self.thickness , 1])

        # Compute Total Transfer Matrix
        layer = 0
        phi = 1j * thickness[0] * self.k[-1][1]
        while layer < layer_num + 1 and np.real(phi) > -100:

            k, R, T = self.fresnel(self.k[-1], refraction[layer], refraction[layer+1])
            self.k.append(k)
            
            self.Tn.append(np.diag([np.exp(phi),np.exp(-phi)]))
            self.Rn.append( np.array([ [1,-R], [-R,1] ])/T )
            self.Mn.append( self.Rn[-1] @ self.Tn[-1] ) 
            
            layer += 1
            phi = 1j * thickness[layer] * self.k[-1][1]
            
        
        # Compute Waves in the Air (Avoid Backward Numerical Amplification)
        layer = 0
        while layer < len(self.Mn) and not np.any(np.isinf(self.Mn[layer])):
            self.M = self.Mn[layer] @ self.M
            layer +=1
        self.wave.append(np.array([1, -self.M[1,0]/self.M[1,1]]))

        # Compute all Waves and Energy Deposited        
        interfaces = np.cumsum(thickness)
        fun_x = np.zeros(len(x))
        layer = 0
        
        while layer < len(self.Mn) - 1 and np.abs(self.wave[-1][0]) > 1e-8:
            # Update Wave at the next Interface (positive side)
            self.wave.append( self.Mn[layer] @ self.wave[-1] )
            # Avoid Numerical Instability
            if abs(self.wave[-1][1]) < 1e-8:
                self.wave[-1][1] = 0
            
            # Compute Dissipation
            index = np.logical_and(x > interfaces[layer], x < interfaces[layer + 1])
            index = np.where(index)
            depth = x[index] - interfaces[layer]
            fun_x[index] = self.div_poynting(self.wave[-1], layer, depth)
            layer += 1
            
        self.wave.append( self.Mn[-1] @ self.wave[-1] )
        fun_x *=  self.wavelength / np.pi / np.cos(self.angle)
        return fun_x

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def fresnel(self, k1, n1, n2):
        
        k0 = 2 * np.pi / self.wavelength
        k2 = np.array([k1[0], np.sqrt( k0**2 * n2**2 - k1[0]**2) ])
        
        if   self.polarization.lower() == 's':
            K1 = k1[1]
            K2 = k2[1]
        elif self.polarization.lower() == 'p':
            K1 = n2**2 * k1[1]
            K2 = n1**2 * k2[1]
        else:
            print("!!! Source Error: Unknown polarization type !!!")
            print("Polarization: " + self.polarization)
            
        R = (K1 - K2) / ( K1 + K2 )
        T = 1 - R
        return k2, R, T 

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def div_poynting(self, EH, layer, x):
        
        k = self.k[layer+1]
        n = self.refraction[layer]
        
        FW = -2*np.imag(k[1])*x
        BW = np.min(np.vstack([FW,np.tile(100,x.size)]),0)
        
        if   self.polarization.lower() == 's':
            Sf = np.abs(EH[0])**2 * np.cos(np.angle(k[1])) * np.exp(FW)
            Sb = np.abs(EH[1])**2 * np.cos(np.angle(k[1])) * np.exp(BW)
            S  = np.abs(k[1]) * np.imag(k[1]) * (Sf+Sb)
        elif self.polarization.lower() == 'p':
            Sf = np.cos(np.angle(k[1]) - 2*np.angle(n)) * np.exp(FW)
            Sb = np.cos(np.angle(k[1]) - 2*np.angle(n)) * np.exp(BW)
            S  = np.abs(EH[0])**2 * Sf + np.abs(EH[1])**2 * Sb
            S *= np.abs(k[1]) * np.imag(k[1]) / np.abs(n**2)
        else:
            print("!!! Source Error: Unknown polarization type !!!")
            print("Polarization: " + self.polarization)
        
        return S

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def setLaser(self, fluence, FWHM):
        self.time = FWHM/np.sqrt(8*np.log(2))
        self.peak = fluence/np.sqrt(2*np.pi*self.time**2)
        self.time_step_hint = FWHM/10