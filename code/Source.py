# =============================================================================
#   TITLE: NTMpy source
# -----------------------------------------------------------------------------
#        Authors: Valentino Scalera, Lukas Alber
#        Version: 1.0
#   Dependencies: numpy, matplotlib, bspline
# -----------------------------------------------------------------------------
import numpy as np

# =============================================================================
# SOURCE CLASS
# =============================================================================
class source(object):
# =============================================================================
#
# -----------------------------------------------------------------------------
    def __init__(self):
        self.type_t       = 'Gaussian'
        self.type_x       = 'LambertBeer'

        self.peak         = 5e10
        self.time         = 2e-12
        self.delay        = 0

        self.polarization = "p"
        self.angle      =  0
        self.refraction = [1]
        self.absorption = [1]

        self.thickness  = [ ]


# =============================================================================
#
# -----------------------------------------------------------------------------
    def getProperties(self): # to depict the properties of the object
        for i in (self.__dict__):
            print(i,' : ',self.__dict__[i])

# =============================================================================
#
# -----------------------------------------------------------------------------
    def __repr__(self):
        return('Source')

# =============================================================================
#
# -----------------------------------------------------------------------------
    def matrix(self, x, t):

        if not isinstance(self.absorption, list):
            self.absorption = [self.absorption]

        xmg,tmg = np.meshgrid(x,t)
        if self.type_t.lower() in ["gaussian","gauss"]:
            fun_t = np.exp(-(tmg-self.delay)**2/(2*self.time**2))
        else:
            print("!!!  Source Error: Unknown time profile  !!!")


        if self.type_x.lower() in ["beerlambert","beer","lambert","lambertbeer","lb"]:
            lam = 1/self.absorption[0]
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

# =============================================================================
#
# -----------------------------------------------------------------------------
    def transfer_matrix(self, x):

        layer_num = len(self.thickness)
        if len(self.refraction) != layer_num or len(self.absorption) != layer_num:
            print("!!! Source Error: Unconsistent number of layer !!!")
        
        self.wave = []
        self.dir  = [self.angle]
        self.Tn = []
        self.Rn = []
        self.Mn = []
        self.M  = np.eye(2)
        
        refraction = np.hstack([1, self.refraction, 1])
        absorption = np.hstack([1, self.absorption, 1])
        thickness  = np.hstack([0, self.thickness , 1])

        l = 0; phi = 0
        while l < layer_num + 1 and np.exp(-phi) > 1e-16:
            
            a, r, t = self.fresnel(self.dir[-1], refraction[l], refraction[l+1])
            self.dir.append(a)
            phi = thickness[l] * np.cos(self.dir[l]) / absorption[l]
            
            self.Tn.append(np.array([ [np.exp(-phi),0], [0,np.exp(phi)] ]))
            self.Rn.append( np.array([ [1,-r], [-r,1] ])/t )
            self.Mn.append( self.Rn[-1] @ self.Tn[-1] ) 
            
            l += 1
        
        l = 0
        while l < len(self.Mn) and not np.any(np.isinf(self.Mn[l])):
            self.M = self.Mn[l] @ self.M
            l +=1
            
        # Compute the Wave in Air
        self.wave.append(np.array([1, -self.M[1,0]/self.M[1,1]]))

        interfaces = np.cumsum(thickness)
        fun_x = np.zeros(len(x))
        sign = np.array([-1,+1])
        counter = 0; l = 0

        while l < layer_num and not np.isnan(self.dir[l]) and self.wave[-1][0] > 1e-9:
            # Update Wave at the next Interface (positive side)
            self.wave.append( self.Mn[l] @ self.wave[-1] )
            # Avoid Numerical Instability
            if abs(self.wave[-1][1]) < 1e-9:
                self.wave[-1][1] = 0
                
            while x[counter] < interfaces[l+1]:
                energy = np.cos(self.dir[l+1]) / self.absorption[l]
                path = ( x[counter] - interfaces[l]) * energy
                fun_x[counter] = np.sum( np.abs(self.wave[-1]) * np.exp( path * sign) * energy)
                counter += 1
            l += 1

        return fun_x


# =============================================================================
#
# -----------------------------------------------------------------------------
    def fresnel(self, th1, n1, n2):
        th2 = np.arcsin( n1/n2 * np.sin(th1) )
        if   self.polarization.lower() == "s":
            r = ( n1*np.cos(th1) - n2*np.cos(th2) ) / ( n1*np.cos(th1) + n2*np.cos(th2) )
        elif self.polarization.lower() == "p":
            r = ( n1*np.cos(th2) - n2*np.cos(th1) ) / ( n1*np.cos(th2) + n2*np.cos(th1) )
        else:
            print("!!! Source Error: Unknown polarization type !!!")
        t = np.sqrt(1-r**2)
        return th2, r, t


# =============================================================================
#
# -----------------------------------------------------------------------------
    def setLaser(self, fluence, FWHM):
        self.time = FWHM/np.sqrt(2*np.log(2))
        self.peak = fluence/np.sqrt(2*np.pi*self.time**2)
