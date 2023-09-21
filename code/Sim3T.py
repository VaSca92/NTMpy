# ========================================================================================
#   TITLE: NTMpy sim3T
# ----------------------------------------------------------------------------------------
#        Authors: Valentino Scalera, Lukas Alber
#        Version: 2.0
#   Dependencies: numpy, matplotlib, bspline, tqdm
# ----------------------------------------------------------------------------------------

import numpy as np
from bspline import Bspline
from bspline.splinelab import aptknt
import time
from tqdm import tqdm

from Source import source

# ========================================================================================
#
# ========================================================================================
class Sim3T(object):

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def __init__(self):
        # Time Data
        self.start_time  = 0  # init. simulation time
        self.final_time  = 0  # final simulation time
        self.time_step   = 0  # simulation time step
        self.dt_ext      = [] # extimated "good" time step
        # Geometric data
        self.x          = np.array([]) # points for spline generation
        self.y          = np.array([]) # points for solution plot
        self.length     = [0] # length of each layer
        self.grd_points = [ ] # number of spline per layer
        self.plt_points = [ ] # numerb of plot points per layer
        self.order      = 5   # order of the splines
        # Electronic System
        self.elec_K  = []  # Thermal Conductivity (electron)
        self.elec_C  = []  # Specific Heat (electron)
        self.elec_QE = []  # derivative of conductivity with elec. temperature
        self.elec_QL = []  # derivative of conductivity with latt. temperature
        self.elec_QS = []  # derivative of condictivity with spin. temperature
        self.LBCT_E  = 1   # Left  Boundary Condition Type  ( 0 = Dirichlet )
        self.RBCT_E  = 1   # Right Boundary Condition Type  ( 1 =   Neumann )
        self.LBCV_E  = 0   # Left  Boundary Condition Value
        self.RBCV_E  = 0   # Right Boundary Condition Value
        self.init_E  = 300 # initial temperature (electron)
        # Lattice System
        self.latt_K  = []  # Thermal Condictivity (lattice)
        self.latt_C  = []  # Specific Heat (lattice)
        self.latt_QE = []  # derivative of conductivity with elec. temperature
        self.latt_QL = []  # derivative of conductivity with latt. temperature
        self.latt_QS = []  # derivative of conductivity with spin. temperature
        self.LBCT_L  = 1   # Left  Boundary Condition Type  ( 0 = Dirichlet )
        self.RBCT_L  = 1   # Right Boundary Condition Type  ( 1 =   Neumann )
        self.LBCV_L  = 0   # Left  Boundary Condition Value
        self.RBCV_L  = 0   # Right Boundary Condition Value
        self.init_L  = 300 # initial temperature (lattice)
        # Spin System
        self.spin_K  = []  # Thermal Condictivity (spin)
        self.spin_C  = []  # Specific Heat (spin)
        self.spin_QE = []  # derivative of conductivity with elec. temperature
        self.spin_QL = []  # derivative of conductivity with latt. temperature
        self.spin_QS = []  # derivative of conductivity with spin. temperature
        self.LBCT_S  = 1   # Left  Boundary Condition Type  ( 0 = Dirichlet )
        self.RBCT_S  = 1   # Right Boundary Condition Type  ( 1 =   Neumann )
        self.LBCV_S  = 0   # Left  Boundary Condition Value
        self.RBCV_S  = 0   # Right Boundary Condition Value
        self.init_S  = 300 # initial temperature (spin)
        # Coupling
        self.GEL = [] # Coupling between electrons and lattice
        self.GES = [] # Coupling between electrons and spin
        self.GLS = [] # Coupling between lattice and spin
        # Differentiation Matrices an Plot Matrix
        self.D0 = np.zeros([0,0]) # Map: spline coeff. -> temperature value (computation)
        self.D1 = np.zeros([0,0]) # Map: spline coeff. -> temperature 1st derivative
        self.D2 = np.zeros([0,0]) # Map: spline coeff. -> temperature 2nd derivative
        self.P0 = np.zeros([0,0]) # Map: spline coeff. -> temperature value (plot)
        self.DL = [] # Map: spline coeff. -> derivative on the left  end of each layer
        self.DR = [] # Map: spline coeff. -> derivative on the right end of each layer
        
        # Source
        self.source = source() # NTMpy source object
        self.sourceset = False # check if source is set
        self.source.peak = 0   # default source is zero
        # Default Settings
        self.default_Ng = 12 # default number of splines per layer
        self.default_Np = 60 # default number of plot points per layer
        # Maximum Temperature Expected (used in time step evaluation)
        self.burn = 2000 # I do not expect temperature to go above this
        

        # Incides of the interfaces/material and number of Layers
        self.layers = 0 # number of layers
        self.interfaceE = [] # nodes where apply interface condition (elec)
        self.diffusionE = [] # nodes where apply diffusion equation  (elec)
        self.interfaceL = [] # nodes where apply interface condition (latt)
        self.diffusionL = [] # nodes where apply diffusion equation  (latt)
        self.interfaceS = [] # nodes where apply interface condition (spin)
        self.diffusionS = [] # nodes where apply diffusion equation  (spin)
        self.layer_ind  = [] # nodes at the layer interfaces
        # Conductivity == 0
        self.zeroE = []  # boolean: layer with zero conductivity (elec)
        self.zeroL = []  # boolean: layer with zero conductivity (latt)
        self.zeroS = []  # boolean: layer with zero conductivity (spin)
        

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
        output  = "==============================================================================================================\n"
        output += " Simulation Object: Diffusion Equation \n"
        output += "==============================================================================================================\n"
        output += 'Number of Temperatures : 2\n'
        output += '\n'
        output += ' Start time : ' + str(self.start_time) + '\n'
        output += ' Final time : ' + str(self.final_time) + '\n'
        output += '  Time step : ' + str(self.time_step) + '\n'
        return output

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def addLayer(self, L, K, C, D, G = [0,0,0], Ng = False, Np = False):
        # Add Layer to the Electron system # # # # # # # # # # # # # # # # # # # # # # # #
        # Thermal Conductivity
        self.elec_K.append(self.lambdize3(K[0], 1, 1))
        # Heat Capacity  (specific heat * density)
        self.elec_C.append(self.lambdize3(C[0], D, 1))
        # Derivative of the thermal conductivity
        dummyQE = self.elec_K[-1]
        self.elec_QE.append(lambda x, y, z: (dummyQE(x+1e-9, y, z)-dummyQE(x, y, z))/1e-9)
        self.elec_QL.append(lambda x, y, z: (dummyQE(x, y+1e-9, z)-dummyQE(x, y, z))/1e-9)
        self.elec_QS.append(lambda x, y, z: (dummyQE(x, y, z+1e-9)-dummyQE(x, y, z))/1e-9)
        # Add Layer to the Lattice system  # # # # # # # # # # # # # # # # # # # # # # # #
        # Thermal Conductivity
        self.latt_K.append(self.lambdize3(K[1], 1, 2))
        # Heat Capacity  (specific heat * density)
        self.latt_C.append(self.lambdize3(C[1], D, 2))
        # Derivative of the thermal conductivity
        dummyQL = self.latt_K[-1]
        self.latt_QE.append(lambda x, y, z: (dummyQL(x+1e-9, y, z)-dummyQL(x, y, z))/1e-9)
        self.latt_QL.append(lambda x, y, z: (dummyQL(x, y+1e-9, z)-dummyQL(x, y, z))/1e-9)
        self.latt_QS.append(lambda x, y, z: (dummyQL(x, y, z+1e-9)-dummyQL(x, y, z))/1e-9)
        # Add Layer to the Spin system   # # # # # # # # # # # # # # # # # # # # # # # # #
        # Thermal Conductivity
        self.spin_K.append(self.lambdize3(K[1], 1, 3))
        # Heat Capacity  (specific heat * density)
        self.spin_C.append(self.lambdize3(C[1], D, 3))
        # Derivative of the thermal conductivity
        dummyQS = self.spin_K[-1]
        self.spin_QE.append(lambda x, y, z: (dummyQS(x+1e-9, y, z)-dummyQS(x, y, z))/1e-9)
        self.spin_QL.append(lambda x, y, z: (dummyQS(x, y+1e-9, z)-dummyQS(x, y, z))/1e-9)
        self.spin_QS.append(lambda x, y, z: (dummyQS(x, y, z+1e-9)-dummyQS(x, y, z))/1e-9)
        # Add Coupling # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.GEL = np.append(self.GEL, G[0])
        self.GES = np.append(self.GES, G[1])
        self.GLS = np.append(self.GLS, G[2])
        # Detect Zeros
        self.zeroE.append(K[0] == 0)
        self.zeroL.append(K[1] == 0)
        self.zeroS.append(K[2] == 0)
        # Add Geometry
        self.length.append(L + self.length[-1])
        Ng = self.default_Ng if not Ng else Ng
        self.grd_points.append(Ng)
        Np = self.default_Np if not Np else Np
        self.plt_points.append(Np)


# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def setSource(self, source, n = 1):
        self.source = source
        self.sourceset = True


# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def lambdize(self, fun, multiplyer = 1 ):
        typecheckN = type(np.array([0])[0])
        typecheckL = type(lambda x: x)
        if isinstance(fun, typecheckL):
            return lambda x: fun(x)*multiplyer
        elif isinstance(fun, (int, float, typecheckN)):
            return lambda x: fun*multiplyer + 0*x


# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def lambdize2(self, fun, multiplyer = 1, arg = 1 ):
        typecheckN = type(np.array([0])[0])
        typecheckL = type(lambda x: x)
        if isinstance(fun, typecheckL):
            try:
                dummy = fun(0,0)
                return lambda x, y: fun(x,y)*multiplyer
            except:
                if   arg == 1:
                    return lambda x, y: fun(x) * multiplyer
                elif arg == 2:
                    return lambda x, y: fun(y) * multiplyer
        elif isinstance(fun, (int, float, typecheckN)):
            return lambda x, y: fun*multiplyer + 0*x


# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def lambdize3(self, fun, multiplyer = 1, arg = 1 ):
        typecheckN = type(np.array([0])[0])
        typecheckL = type(lambda x: x)
        if isinstance(fun, typecheckL):
            try:
                dummy = fun(0,0,0)
                return lambda x, y, z: fun(x,y,z)*multiplyer
            except:
                if   arg == 1:
                    return lambda x, y, z: fun(x) * multiplyer
                elif arg == 2:
                    return lambda x, y, z: fun(y) * multiplyer
                elif arg == 3:
                    return lambda x, y, z: fun(z) * multiplyer
        elif isinstance(fun, (int, float, typecheckN)):
            return lambda x, y, z: fun*multiplyer + 0*x

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def build_geometry(self):
        # Store layer and interface number
        self.layers = len(self.length) - 1
        # Indices of the Interface points
        IFg  = np.append( 0, np.cumsum(self.grd_points) )
        IFg -= np.arange( IFg.size )
        IFp  = np.append( 0, np.cumsum(self.plt_points) )
        IFp -= np.arange( IFp.size )
        # Preallocate the Differentation Matrices
        self.D0 = np.zeros([IFg[-1] + 1, IFg[-1] + 1])
        self.D1 = np.zeros([IFg[-1] + 1, IFg[-1] + 1])
        self.D2 = np.zeros([IFg[-1] + 1, IFg[-1] + 1])
        self.P0 = np.zeros([IFp[-1] + 1, IFg[-1] + 1])
        # Select Order of the Spline (future expert setting)
        order = self.order
        # Layer Stability Matrix
        LSM = []
        # FOR each layer: Construct the Ai Matrices
        for i in range( len(self.length) - 1):
            # Define Space Points
            x = np.linspace(self.length[i], self.length[i+1] , self.grd_points[i])
            y = np.linspace(self.length[i], self.length[i+1] , self.plt_points[i])
            # Spline Generation
            knot_vector = aptknt( x, order)
            basis       = Bspline(knot_vector, order)
            # Generate Differentiation and Plot Matrices for the Layer
            D0L = basis.collmat(x, deriv_order = 0)
            D1L = basis.collmat(x, deriv_order = 1)
            D2L = basis.collmat(x, deriv_order = 2)
            P0L = basis.collmat(y, deriv_order = 0)
            # Correct BSpline Package Bug
            D0L[-1,-1] = 1; P0L[-1,-1] = 1;
            D1L[-1] = -np.flip(D1L[0],0)
            D2L[-1] = +np.flip(D2L[0],0)
            # Matrices for the Stablity
            LSM.append(np.array( D2L @ np.linalg.inv(D0L)))
            # Total Differentiation and Plot Matrices
            self.D0[ IFg[i] : IFg[i+1] + 1, IFg[i] : IFg[i+1] + 1] = D0L
            self.D1[ IFg[i] : IFg[i+1] + 1, IFg[i] : IFg[i+1] + 1] = D1L
            self.D2[ IFg[i] : IFg[i+1] + 1, IFg[i] : IFg[i+1] + 1] = D2L
            self.P0[ IFp[i] : IFp[i+1] + 1, IFg[i] : IFg[i+1] + 1] = P0L
            # Non derivable points
            self.D1[ IFg[i+1], IFg[i+1]] = 0
            self.D2[ IFg[i+1], IFg[i+1]] = 0
            # Derivative from Left and Right
            self.DL.append(D1L[ 0])
            self.DR.append(D1L[-1])
            # Extend the Space Grid
            self.x = np.append( self.x[:-1], x)
            self.y = np.append( self.y[:-1], y)
        self.interfaceE = IFg
        self.interfaceL = IFg
        self.interfaceS = IFg
        self.layer_ind  = np.vstack([ IFg[:-1] + 1, IFg[1:]]).transpose()
        self.diffusionE = self.layer_ind.copy()
        self.diffusionL = self.layer_ind.copy()
        self.diffusionS = self.layer_ind.copy()
        self.detect_id()
        # Calculate approximated Time steps
        self.dt_ext = self.stability(LSM)


# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def detect_id(self):
        
        for i in range( len(self.zeroE) - 2, -1, -1):
            if self.zeroE[i] and self.zeroE[i+1]:
                self.interfaceE = np.delete(self.interfaceE, i)
                self.diffusionE[ i+1, 0] -= 1
        
        if self.zeroE[ 0]:
            self.diffusionE[ 0, 0] -= 1
        if self.zeroE[-1]:
            self.diffusionE[-1,-1] += 1
        
        
        for i in range( len(self.zeroL) - 2, -1, -1):
            if self.zeroL[i] and self.zeroL[i+1]:
                self.interfaceL = np.delete(self.interfaceL, i)
                self.diffusionL[ i+1, 0] -= 1
                
        if self.zeroL[ 0]:
            self.diffusionL[ 0, 0] -= 1
        if self.zeroL[-1]:
            self.diffusionL[ 0, 0] += 1


        for i in range( len(self.zeroS) - 2, -1, -1):
            if self.zeroS[i] and self.zeroS[i+1]:
                self.interfaceS = np.delete(self.interfaceS, i)
                self.diffusionS[ i+1, 0] -= 1
                
        if self.zeroS[ 0]:
            self.diffusionS[ 0, 0] -= 1
        if self.zeroS[-1]:
            self.diffusionS[ 0, 0] += 1
            
# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def generate_BC(self):
        # Constant for Type Checking
        typecheck = type(lambda x: x)
        # Initialization
        BC_E = np.zeros([2 ,len(self.t)])
        BC_L = np.zeros([2 ,len(self.t)])
        BC_S = np.zeros([2 ,len(self.t)])
        # Create Boundary Condition for the Electronic System
        if isinstance(self.LBCV_E, typecheck): BC_E[0] = self.LBCV_E(self.t)
        else: BC_E[0] = np.tile(self.LBCV_E, len(self.t))
        if isinstance(self.RBCV_E, typecheck): BC_E[1] = self.RBCV_E(self.t)
        else: BC_E[1] = np.tile(self.RBCV_E, len(self.t))
        # Create Boundary Condition for the Lattice System
        if isinstance(self.LBCV_L, typecheck): BC_L[0] = self.LBCV_L(self.t)
        else: BC_L[0] = np.tile(self.LBCV_L, len(self.t))
        if isinstance(self.RBCV_L, typecheck): BC_L[1] = self.RBCV_L(self.t)
        else: BC_L[1] = np.tile(self.RBCV_L, len(self.t))
        # Create Boundary Condition for the Spin System
        if isinstance(self.LBCV_S, typecheck): BC_S[0] = self.LBCV_S(self.t)
        else: BC_S[0] = np.tile(self.LBCV_S, len(self.t))
        if isinstance(self.RBCV_S, typecheck): BC_S[1] = self.RBCV_S(self.t)
        else: BC_S[1] = np.tile(self.RBCV_S, len(self.t))
        
        # Return Boundary Condition
        return BC_E, BC_L, BC_S

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def generate_init(self):
        # Constant for Type Checking
        typecheck = type(lambda x: x)
        # Set initial Condition
        if isinstance(self.init_E, typecheck):
            ce = np.linalg.solve(self.D0, self.init_E(self.x))
            ue = self.init_E(self.y)
        elif isinstance(self.init_E, (int, float)):
            ce = np.tile(self.init_E, len(self.x))
            ue = np.tile(self.init_E, len(self.y))
        
        if isinstance(self.init_L, typecheck):
            cl = np.linalg.solve(self.D0, self.init_L(self.x))
            ul = self.init_L(self.y)
        elif isinstance(self.init_L, (int, float)):
            cl = np.tile(self.init_L, len(self.x))
            ul = np.tile(self.init_L, len(self.y))
        
        if isinstance(self.init_S, typecheck):
            cs = np.linalg.solve(self.D0, self.init_S(self.x))
            us = self.init_S(self.y)
        elif isinstance(self.init_S, (int, float)):
            cs = np.tile(self.init_S, len(self.x))
            us = np.tile(self.init_S, len(self.y))
        
        
        return ce, ue, cl, ul, cs, us

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def generate_matrix(self):
        # Matrics of Coefficient fot the Electron and Lattice System
        LHSE = self.D0.copy(); LHSL = self.D0.copy(); LHSS = self.D0.copy();
        
        # Setting Boundary Condition Type
        if self.LBCT_E == 1 and not self.zeroE[ 0]:
            LHSE[ 0, :self.grd_points[ 0] ] = -self.DL[ 0].copy()
        if self.RBCT_E == 1 and not self.zeroE[-1]:
            LHSE[-1, -self.grd_points[-1]:] =  self.DR[-1].copy()
            
        if self.LBCT_L == 1 and not self.zeroL[ 0]:
            LHSL[ 0, :self.grd_points[0] ] = -self.DL[ 0].copy()
        if self.RBCT_L == 1 and not self.zeroL[-1]:
            LHSL[-1, -self.grd_points[-1]:] =  self.DR[-1].copy()
        
        if self.LBCT_S == 1 and not self.zeroS[ 0]:
            LHSS[ 0, :self.grd_points[0] ] = -self.DL[ 0].copy()
        if self.RBCT_S == 1 and not self.zeroS[-1]:
            LHSS[-1, -self.grd_points[-1]:] =  self.DR[-1].copy()
        
        LHSE[self.interfaceE[+1:-1]] = 0;
        LHSL[self.interfaceL[+1:-1]] = 0;
        LHSS[self.interfaceL[+1:-1]] = 0;
        
        return LHSE, LHSL, LHSS

# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def run(self):
        if not self.sourceset:
            self.warning(4)
        # --------------------------------------------------------------------------------
        # Setup Phase: Timestep evaluation, Source Generation,
        #              Boundary Condition prepared, Adjust Coupling
        # --------------------------------------------------------------------------------
        self.build_geometry()
        # Load all the Geometry Matrices -------------------------------------------------
        # Rename some Instance Variables
        BCEL = not self.zeroE[0]; BCER = not self.zeroE[-1]
        BCLL = not self.zeroL[0]; BCLR = not self.zeroL[-1]
        BCLS = not self.zeroS[0]; BCLS = not self.zeroS[-1]
        # STABILITY EVALUATION ###########################################################
        # Calculating the preferred time step
        idealtimestep  = np.min(self.dt_ext)
        # Warnings for missing or bad time step !!!
        if not self.time_step:
            self.time_step  = idealtimestep
            self.warning(1, str(idealtimestep))
        if (self.time_step - idealtimestep)/idealtimestep > 0.5:
            self.warning(2, str(self.time_step), str(idealtimestep))
        if(self.time_step - idealtimestep)/idealtimestep < -0.5:
            self.warning(3, str(self.time_step), str(idealtimestep))
        if self.final_time > 1: self.final_time = self.final_time*self.time_step
        # Define the time vector
        self.t = np.arange(self.start_time, self.final_time, self.time_step)
        # Generate all the matrices ######################################################
        LHSE, LHSL, LHSS             = self.generate_matrix()
        BC_E, BC_L, BC_S             = self.generate_BC()
        c_E, u_E, c_L, u_L, c_S, u_S = self.generate_init()
        # SOURCE GENERATION ##############################################################
        self.source.thickness = np.diff(self.length)
        source = self.source.create(self.x, self.t)
        # ------------------------------------------- Setup ended ------------------------

        # --------------------------------------------------------------------------------
        #  MAIN LOOP
        # --------------------------------------------------------------------------------
        # Initialization of the variables for the Electronic System
        phi_E    = np.zeros([len(self.t),len(self.y)])
        Flow_0E  = np.zeros( len(self.x) )
        Flow_1E  = np.zeros( len(self.x) )
        Flow_2E  = np.zeros( len(self.x) )
        Flow_3E  = np.zeros( len(self.x) )
        dphi_E   = np.zeros( len(self.x) )
        RHSE     = np.zeros( len(self.x) )
        #Initialization of the variables for the Lattice System
        phi_L    = np.zeros([len(self.t),len(self.y)])
        Flow_0L  = np.zeros( len(self.x) )
        Flow_1L  = np.zeros( len(self.x) )
        Flow_2L  = np.zeros( len(self.x) )
        Flow_3L  = np.zeros( len(self.x) )
        dphi_L   = np.zeros( len(self.x) )
        RHSL     = np.zeros( len(self.x) )
        #Initialization of the variables for the Spin System
        phi_S    = np.zeros([len(self.t),len(self.y)])
        Flow_0S  = np.zeros( len(self.x) )
        Flow_1S  = np.zeros( len(self.x) )
        Flow_2S  = np.zeros( len(self.x) )
        Flow_3S  = np.zeros( len(self.x) )
        dphi_S   = np.zeros( len(self.x) )
        RHSS     = np.zeros( len(self.x) )
        # Set Initial Condition
        phi_E[0] = u_E; phi_L[0] = u_L; phi_S[0] = u_S 
        # Identify Layer
        LI = self.layer_ind
        DE = self.diffusionE
        DL = self.diffusionL
        DS = self.diffusionS
        # HERE STARTS THE MAIN LOOP
        start_EL = time.time()
        for i in tqdm(range(1,len(self.t))):
            # Go from coefficient c to phi and its derivatives
            phi0_E = self.D0 @ c_E; phi0_L = self.D0 @ c_L; phi0_S = self.D0 @ c_S
            phi1_E = self.D1 @ c_E; phi1_L = self.D1 @ c_L; phi1_S = self.D1 @ c_S
            phi2_E = self.D2 @ c_E; phi2_L = self.D2 @ c_L; phi2_S = self.D2 @ c_S
            for j in range(len(DE)):  # For every Layer
                # Conduction Heat Flow in the Electronic System
                Flow_0E[DE[j,0]:DE[j,1]] = self.elec_K [j](phi0_E[DE[j,0]:DE[j,1]], phi0_L[DE[j,0]:DE[j,1]], phi0_S[DE[j,0]:DE[j,1]])
                Flow_1E[DE[j,0]:DE[j,1]] = self.elec_QE[j](phi0_E[DE[j,0]:DE[j,1]], phi0_L[DE[j,0]:DE[j,1]], phi0_S[DE[j,0]:DE[j,1]])
                Flow_2E[DE[j,0]:DE[j,1]] = self.elec_QL[j](phi0_E[DE[j,0]:DE[j,1]], phi0_L[DE[j,0]:DE[j,1]], phi0_S[DE[j,0]:DE[j,1]])
                Flow_3E[DE[j,0]:DE[j,1]] = self.elec_QS[j](phi0_E[DE[j,0]:DE[j,1]], phi0_L[DE[j,0]:DE[j,1]], phi0_S[DE[j,0]:DE[j,1]])
                Flow_0E[DE[j,0]:DE[j,1]] *= phi2_E[DE[j,0]:DE[j,1]]
                Flow_1E[DE[j,0]:DE[j,1]] *= phi1_E[DE[j,0]:DE[j,1]]**2
                Flow_2E[DE[j,0]:DE[j,1]] *= phi1_E[DE[j,0]:DE[j,1]]*phi1_L[DE[j,0]:DE[j,1]]
                Flow_3E[DE[j,0]:DE[j,1]] *= phi1_E[DE[j,0]:DE[j,1]]*phi1_S[DE[j,0]:DE[j,1]]
                # Conduction Heat Flow in the Lattice System
                Flow_0L[DL[j,0]:DL[j,1]] = self.latt_K [j](phi0_E[DL[j,0]:DL[j,1]], phi0_L[DL[j,0]:DL[j,1]], phi0_S[DL[j,0]:DL[j,1]])
                Flow_1L[DL[j,0]:DL[j,1]] = self.latt_QE[j](phi0_E[DL[j,0]:DL[j,1]], phi0_L[DL[j,0]:DL[j,1]], phi0_S[DL[j,0]:DL[j,1]])
                Flow_2L[DL[j,0]:DL[j,1]] = self.latt_QL[j](phi0_E[DL[j,0]:DL[j,1]], phi0_L[DL[j,0]:DL[j,1]], phi0_S[DL[j,0]:DL[j,1]])
                Flow_3L[DL[j,0]:DL[j,1]] = self.latt_QS[j](phi0_E[DL[j,0]:DL[j,1]], phi0_L[DL[j,0]:DL[j,1]], phi0_S[DL[j,0]:DL[j,1]])
                Flow_0L[DL[j,0]:DL[j,1]] *= phi2_L[DL[j,0]:DL[j,1]]
                Flow_1L[DL[j,0]:DL[j,1]] *= phi1_L[DL[j,0]:DL[j,1]]*phi1_L[DL[j,0]:DL[j,1]]
                Flow_2L[DL[j,0]:DL[j,1]] *= phi1_L[DL[j,0]:DL[j,1]]**2
                Flow_3L[DL[j,0]:DL[j,1]] *= phi1_L[DL[j,0]:DL[j,1]]*phi1_S[DL[j,0]:DL[j,1]]
                # Conduction Heat Flow in the Spin System
                Flow_0S[DS[j,0]:DS[j,1]] = self.spin_K [j](phi0_E[DS[j,0]:DS[j,1]], phi0_L[DS[j,0]:DS[j,1]], phi0_S[DS[j,0]:DS[j,1]])
                Flow_1S[DS[j,0]:DS[j,1]] = self.spin_QE[j](phi0_E[DS[j,0]:DS[j,1]], phi0_L[DS[j,0]:DS[j,1]], phi0_S[DS[j,0]:DS[j,1]])
                Flow_2S[DS[j,0]:DS[j,1]] = self.spin_QL[j](phi0_E[DS[j,0]:DS[j,1]], phi0_L[DS[j,0]:DS[j,1]], phi0_S[DS[j,0]:DS[j,1]])
                Flow_3S[DS[j,0]:DS[j,1]] = self.spin_QS[j](phi0_E[DS[j,0]:DS[j,1]], phi0_L[DS[j,0]:DS[j,1]], phi0_S[DS[j,0]:DS[j,1]])
                Flow_0S[DS[j,0]:DS[j,1]] *= phi2_S[DS[j,0]:DS[j,1]]
                Flow_1S[DS[j,0]:DS[j,1]] *= phi1_E[DS[j,0]:DS[j,1]]*phi1_S[DS[j,0]:DS[j,1]]
                Flow_2S[DS[j,0]:DS[j,1]] *= phi1_L[DS[j,0]:DS[j,1]]*phi1_S[DS[j,0]:DS[j,1]]
                Flow_3S[DS[j,0]:DS[j,1]] *= phi1_S[DS[j,0]:DS[j,1]]**2
                
                # Diffusion Equation
                dphi_E[DE[j,0]:DE[j,1]] = Flow_0E[DE[j,0]:DE[j,1]] + Flow_1E[DE[j,0]:DE[j,1]] + Flow_2E[DE[j,0]:DE[j,1]] + self.GEL[j]*(phi0_L[DE[j,0]:DE[j,1]]-phi0_E[DE[j,0]:DE[j,1]]) + self.GES[j]*(phi0_S[DE[j,0]:DE[j,1]]-phi0_E[DE[j,0]:DE[j,1]]) + source[DE[j,0]:DE[j,1],i]
                dphi_L[DL[j,0]:DL[j,1]] = Flow_0L[DL[j,0]:DL[j,1]] + Flow_1L[DL[j,0]:DL[j,1]] + Flow_2L[DL[j,0]:DL[j,1]] + self.GEL[j]*(phi0_E[DL[j,0]:DL[j,1]]-phi0_L[DL[j,0]:DL[j,1]]) + self.GLS[j]*(phi0_S[DL[j,0]:DL[j,1]]-phi0_L[DL[j,0]:DL[j,1]])
                dphi_S[DS[j,0]:DS[j,1]] = Flow_0S[DS[j,0]:DS[j,1]] + Flow_1S[DS[j,0]:DS[j,1]] + Flow_2S[DS[j,0]:DS[j,1]] + self.GES[j]*(phi0_E[DS[j,0]:DS[j,1]]-phi0_S[DS[j,0]:DS[j,1]]) + self.GLS[j]*(phi0_L[DS[j,0]:DS[j,1]]-phi0_S[DS[j,0]:DS[j,1]])
                dphi_E[DE[j,0]:DE[j,1]] /= self.elec_C[j](phi0_E, phi0_L, phi0_S)[DE[j,0]:DE[j,1]]
                dphi_L[DL[j,0]:DL[j,1]] /= self.latt_C[j](phi0_E, phi0_L, phi0_S)[DL[j,0]:DL[j,1]]
                dphi_S[DS[j,0]:DS[j,1]] /= self.spin_C[j](phi0_E, phi0_L, phi0_S)[DS[j,0]:DS[j,1]]
            # Apply Heat Conservation on surfaces
            for k, j in enumerate(self.interfaceE[1:-1]): # For every interface
                # Calculate the Flux into and out from the interface
                IFconL = self.elec_K[ k ](phi0_E[j], phi0_L[j], phi0_S[j])*self.DR[ k ]
                IFconR = self.elec_K[k+1](phi0_E[j], phi0_L[j], phi0_S[j])*self.DL[k+1]
                # Electronic System
                LHSE[j, LI[ k ,0] - 1 : LI[ k ,1]] = -IFconL[:-1]
                LHSE[j, LI[k+1,0] : LI[k+1,1] + 1] = +IFconR[+1:]
                LHSE[j, j] = IFconR[0] - IFconL[-1]
            
            for k, j in enumerate(self.interfaceL[1:-1]):
                # Calculate the Flux into and out from the interface
                IFconL = self.latt_K[ k ](phi0_E[j], phi0_L[j], phi0_S[j])*self.DR[ k ]
                IFconR = self.latt_K[k+1](phi0_E[j], phi0_L[j], phi0_S[j])*self.DL[k+1]
                # Lattice System
                LHSL[j, LI[ k ,0] - 1 : LI[ k ,1]] = -IFconL[:-1]
                LHSL[j, LI[k+1,0] : LI[k+1,1] + 1] = +IFconR[+1:]
                LHSL[j, j] = IFconR[0] - IFconL[-1]
                
            for k, j in enumerate(self.interfaceS[1:-1]): # For every interface
                # Calculate the Flux into and out from the interface
                IFconL = self.spin_K[ k ](phi0_E[j], phi0_L[j], phi0_S[j])*self.DR[ k ]
                IFconR = self.spin_K[k+1](phi0_E[j], phi0_L[j], phi0_S[j])*self.DL[k+1]
                # Electronic System
                LHSS[j, LI[ k ,0] - 1 : LI[ k ,1]] = -IFconL[:-1]
                LHSS[j, LI[k+1,0] : LI[k+1,1] + 1] = +IFconR[+1:]
                LHSS[j, j] = IFconR[0] - IFconL[-1]
            # Applying Explicit Euler Method
            RHSE = phi0_E + self.time_step * dphi_E
            RHSL = phi0_L + self.time_step * dphi_L
            RHSS = phi0_S + self.time_step * dphi_S
            # Make Room for Boundary Condition and Interface Condition
            if BCEL: RHSE[ 0] = BC_E[0,i]/self.elec_K[ 0](phi0_E[ 0], phi0_L[ 0], phi0_S[ 0])**self.LBCT_E
            if BCER: RHSE[-1] = BC_E[1,i]/self.elec_K[-1](phi0_E[-1], phi0_L[-1], phi0_S[-1])**self.RBCT_E
            if BCLL: RHSL[ 0] = BC_L[0,i]/self.latt_K[ 0](phi0_E[ 0], phi0_L[ 0], phi0_S[ 0])**self.LBCT_L
            if BCLR: RHSL[-1] = BC_L[1,i]/self.latt_K[-1](phi0_E[-1], phi0_L[-1], phi0_S[-1])**self.RBCT_L
            if BCLS: RHSS[ 0] = BC_S[0,i]/self.spin_K[ 0](phi0_E[ 0], phi0_L[ 0], phi0_S[ 0])**self.LBCT_L
            if BCLS: RHSS[-1] = BC_S[1,i]/self.spin_K[-1](phi0_E[-1], phi0_L[-1], phi0_S[-1])**self.RBCT_L
            
            # Calculate the new value of the Temperature
            c_E = np.linalg.solve(LHSE,RHSE)
            c_L = np.linalg.solve(LHSL,RHSL)
            c_S = np.linalg.solve(LHSS,RHSS)
            # Store The Temperature on the refined grid in a variable
            phi_E[i] = self.P0 @ c_E; phi_L[i] = self.P0 @ c_L; phi_S[i] = self.P0 @ c_S;
        # END OF THE MAIN LOOP
        end_EL = time.time()
        self.warning(0, str(end_EL - start_EL))
        return self.y, self.t, np.transpose(np.dstack([phi_E, phi_L, phi_S]))


# ========================================================================================
#
# ----------------------------------------------------------------------------------------
    def stability(self, LSM):
        # Useful Constant
        test       = np.linspace(270, self.burn, 50)
        eigs       = np.zeros([len(LSM)])
        for i in range(self.layers):
            dim = len(LSM[i]); DIF = np.zeros([3*dim,3*dim])
            # Worst case for the Diffusion Instability
            DE = max(self.elec_K[i](test, test, test)/self.elec_C[i](test, test, test))
            DL = max(self.latt_K[i](test, test, test)/self.latt_C[i](test, test, test))
            DS = max(self.spin_K[i](test, test, test)/self.spin_C[i](test, test, test))
            # Worst case for the Coupling Instability
            XEL = max(self.GEL[i]/self.elec_C[i](test, test, test))
            XLE = max(self.GEL[i]/self.latt_C[i](test, test, test))
            XES = max(self.GES[i]/self.elec_C[i](test, test, test))
            XSE = max(self.GES[i]/self.spin_C[i](test, test, test))
            XLS = max(self.GLS[i]/self.latt_C[i](test, test, test))
            XSL = max(self.GLS[i]/self.spin_C[i](test, test, test))
            # Instability due to Diffusion
            DIF  = np.kron(np.diag([DE,DL,DS]), LSM[i])
            # Instability due to Coupling
            EXC = np.array([[-XEL-XES,XEL,XES],[XLE,-XLE-XLS,XLS],[XSE,XSL,-XSE-XSL]])
            EXC = np.kron( EXC, np.eye(dim))
            # Total Instability
            StbMat = EXC + DIF
            # Evaluate the Eigenvalues
            eigs[i] = min(np.real(np.linalg.eig(StbMat)[0]))
            min_exc = min(-XEL, -XLE, -XES, -XSE, -XLS, -XSL)
            eigs[i] = min(eigs[i], min_exc/.5)
        # Return the smallest time step
        return min(-1.9/eigs)

# ========================================================================================
#
# ========================================================================================
    def warning(self, msg, arg1 = '*missing*', arg2 = '*missing*'):
        if msg == 0:
            text = \
            ' Heat diffusion in a coupled electron-lattice system has been simulated! \n' + \
            ' Eleapsed time in E.E.- loop: ' + arg1 + '\n'
        if msg == 1:
            text = \
            ' No specific time constant has been indicated. \n' + \
            ' The stability region has been calculated and an appropriate timestep has been chosen.\n' + \
            ' Timestep = ' + arg1 + ' s\n'
        if msg == 2:
            text = \
            ' The manually chosen time step of ' + arg1 + ' is eventually too big and could cause instabilities in the simulation.\n' + \
            ' We suggest a timestep of ' + arg2 + ' s\n'
        if msg == 3:
            text = \
            ' The maunually chosen time step of ' + arg1 + ' is very small and will eventually cause a long simulation time.\n' + \
            ' We suggest a timestep of' + arg2 + ' s\n'
        if msg == 4:
            text = \
            ' The source is not set, please recontrol the code. The command is sim.setsource(source)\n'
        print(\
    '--------------------------------------------------------------------------------------------------------\n' + \
    text + \
    '--------------------------------------------------------------------------------------------------------\n')

