# =============================================================================
#   TITLE: NTMpy sim2T
# -----------------------------------------------------------------------------
#        Authors: Valentino Scalera, Lukas Alber
#        Version: 2.0
#   Dependencies: numpy, matplotlib, bspline
# -----------------------------------------------------------------------------

import numpy as np
from bspline import Bspline
from bspline.splinelab import aptknt
import time
from tqdm import tqdm

from Source import source

# =============================================================================
#
# =============================================================================
class Sim2T1L(object):

# =============================================================================
#
# -----------------------------------------------------------------------------
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
        self.plt_points = [ ] # number of plot points per layer
        self.order      = 5   # order of the splines
        # Electronic System
        self.elec_K  = []  # Thermal Conductivity (electron)
        self.elec_C  = []  # Specific Heat (electron)
        self.elec_QE = []  # derivative of conductivity with elec. temperature
        self.elec_QL = []  # derivative of conductivity with latt. temperature
        self.init_E  = 300 # initial temperature (electron)
        # Lattice System
        self.latt_K  = []  # Thermal Conductivity (lattice)
        self.latt_C  = []  # Specific Heat (lattice)
        self.latt_QE = []  # derivative of conductivity with elec. temperature
        self.latt_QL = []  # derivative of conductivity with latt. temperature
        self.init_L  = 300 # initial temperature (lattice)
        # Coupling constant
        self.G = []
        
        # Differentiation Matrices an Plot Matrix
        self.D0 = np.zeros([0,0]) # Map: spline coeff. -> temperature value (computation)
        self.D1 = np.zeros([0,0]) # Map: spline coeff. -> temperature 1st derivative
        self.D2 = np.zeros([0,0]) # Map: spline coeff. -> temperature 2nd derivative
        self.P0 = np.zeros([0,0]) # Map: spline coeff. -> temperature value (plot)
        self.I0 = np.zeros([0,0])
        # Source
        self.source = source() # NTMpy source object
        self.sourceset = False # check if source is set
        self.source.peak = 0   # default source is zero
        # Default Settings
        self.default_Ng =  12 # default number of splines per layer
        self.default_Np = 100 # default number of plot points per layer
        # Maximum Temperature Expected (used in time step evaluation)
        self.burn = 7000 # I do not expect temperature to go above this
        
        self.debug = np.array([])

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
        output  = "==============================================================================================================\n"
        output += " Simulation Object: Diffusion Equation \n"
        output += "==============================================================================================================\n"
        output += 'Number of Temperatures : 2\n'
        output += '\n'
        output += ' Start time : ' + str(self.start_time) + '\n'
        output += ' Final time : ' + str(self.final_time) + '\n'
        output += '  Time step : ' + str(self.time_step) + '\n'
        return output

# =============================================================================
#
# -----------------------------------------------------------------------------
    def setMaterial(self, L, K, C, D, G = 0, Ng = False, Np = False):
        # Add Layer to the Electron system # # # # # # # # # # # # # # # # # #
        # Thermal Conductivity
        self.elec_K = self.lambdize2(K[0], 1, 1)
        # Heat Capacity  (specific heat * density)
        self.elec_C = self.lambdize2(C[0], D, 1)
        # Derivative of the thermal conductivity
        dummyQE = self.elec_K
        self.elec_QE = lambda x, y: (dummyQE(x+1e-9, y)-dummyQE(x, y))/1e-9
        self.elec_QL = lambda x, y: (dummyQE(x, y+1e-9)-dummyQE(x, y))/1e-9
        # Add Layer to the Lattice system  # # # # # # # # # # # # # # # # # #
        # Thermal Conductivity
        self.latt_K = self.lambdize2(K[-1], 1, 2)
        # Heat Capacity  (specific heat * density)
        self.latt_C = self.lambdize2(C[-1], D, 2)
        # Derivative of the thermal conductivity
        dummyQL = self.latt_K
        self.latt_QE = lambda x, y: (dummyQL(x+1e-9, y)-dummyQL(x, y))/1e-9
        self.latt_QL = lambda x, y: (dummyQL(x, y+1e-9)-dummyQL(x, y))/1e-9
        # Add Coupling # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.G = self.lambdize2(G, 1, 1)
        # Detect Zeros
        self.zeroE = K[ 0] == 0
        self.zeroL = K[-1] == 0
        # Add Geometry
        self.length = L
        self.grd_points = self.default_Ng if not Ng else Ng
        self.plt_points = self.default_Np if not Np else Np


# =============================================================================
#
# -----------------------------------------------------------------------------
    def setSource(self, source, n = 1):
        self.source = source
        self.sourceset = True


# =============================================================================
#
# -----------------------------------------------------------------------------
    def lambdize(self, fun, multiplyer = 1 ):
        typecheckN = type(np.array([0])[0])
        typecheckL = type(lambda x: x)
        if isinstance(fun, typecheckL):
            return lambda x: fun(x)*multiplyer
        elif isinstance(fun, (int, float, typecheckN)):
            return lambda x: fun*multiplyer + 0*x


# =============================================================================
#
# -----------------------------------------------------------------------------
    def lambdize2(self, fun, multiplyer = 1, arg = 1 ):
        typecheckN = type(np.array([0])[0])
        typecheckL = type(lambda x: x)
        if isinstance(fun, typecheckL):
            try:
                dummy = fun(1,1)
                return lambda x, y: fun(x,y)*multiplyer
            except:
                if   arg == 1:
                    return lambda x, y: fun(x) * multiplyer
                elif arg == 2:
                    return lambda x, y: fun(y) * multiplyer
        elif isinstance(fun, (int, float, typecheckN)):
            return lambda x, y: fun*multiplyer + 0*x

# =============================================================================
#
# -----------------------------------------------------------------------------
    def build_geometry(self):
        # Switch to Instance Variables
        length     = self.length
        plt_points = self.plt_points
        grd_points = self.grd_points
        # Select Order of the Spline (future expert setting)
        order = self.order
        # Construct the Di Matrices
        x = np.linspace( 0, length, grd_points)
        y = np.linspace( 0, length, plt_points)
        self.x = x; self.y = y
        # Spline Generation
        knot_vector = aptknt( x, order)
        basis       = Bspline(knot_vector, order)
        # Generate Differentiation and Plot Matrices for the Layer
        self.D0 = basis.collmat(x, deriv_order = 0)
        self.D1 = basis.collmat(x, deriv_order = 1)
        self.D2 = basis.collmat(x, deriv_order = 2)
        self.P0 = basis.collmat(y, deriv_order = 0)
        # Correct BSpline Package Bug
        self.D0[-1,-1] = 1; self.P0[-1,-1] = 1
        A = self.D0.copy()
        A[ 0,:] = self.D1[0,:].copy()
        A[-1,:] = np.flip(A[0,:])
        self.I0 = np.linalg.inv(A)
        # Matrices for the Stablity
        LSM = self.D2 @ self.I0
        # Calculate approximated Time steps
        self.dt_ext = self.stability(LSM)


# =============================================================================
#
# -----------------------------------------------------------------------------
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
        return ce, ue, cl, ul


# =============================================================================
#
# -----------------------------------------------------------------------------
    def run(self):
        
        if not self.sourceset:
            self.warning(4)
        # ---------------------------------------------------------------------
        # Setup Phase: Timestep evaluation, Source Generation,
        #              Boundary Condition prepared, Adjust Coupling
        # ---------------------------------------------------------------------
        self.build_geometry()
        # Load all the Geometry Matrices --------------------------------------
        # STABILITY EVALUATION ################################################
        # Calculating the preferred time step
        idealtimestep  = np.min(self.dt_ext)
        idealtimestep  = np.min([idealtimestep, self.source.time_step_hint])
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
        # Generate all the matrices ###########################################
        c_E, u_E, c_L, u_L = self.generate_init()
        # SOURCE GENERATION ###################################################
        self.source.thickness = [self.length]
        source = self.source.matrix(self.x, self.t)
        # ------------------------------------------- Setup ended -------------

        # ---------------------------------------------------------------------
        #  MAIN LOOP
        # ---------------------------------------------------------------------
        # Initialization of the variables for the Electronic System
        phi_E    = np.zeros([len(self.t),len(self.y)])
        Flow_0E  = np.zeros( len(self.x) )
        Flow_1E  = np.zeros( len(self.x) )
        Flow_2E  = np.zeros( len(self.x) )
        dphi_E   = np.zeros( len(self.x) )
        #Initialization of the variables for the Lattice System
        phi_L    = np.zeros([len(self.t),len(self.y)])
        Flow_0L  = np.zeros( len(self.x) )
        Flow_1L  = np.zeros( len(self.x) )
        Flow_2L  = np.zeros( len(self.x) )
        dphi_L   = np.zeros( len(self.x) )
        # Set Initial Condition
        phi_E[0] = u_E; phi_L[0] = u_L
        # HERE STARTS THE MAIN LOOP
        start_EL = time.time()
        
        self.debug = np.zeros([2,len(self.t)])
        
        for i in tqdm(range(1,len(self.t))):
            # Go from coefficient c to phi and its derivatives
            phi0_E = self.D0[1:-1,:] @ c_E; phi0_L = self.D0[1:-1,:] @ c_L
            phi1_E = self.D1[1:-1,:] @ c_E; phi1_L = self.D1[1:-1,:] @ c_L
            phi2_E = self.D2[1:-1,:] @ c_E; phi2_L = self.D2[1:-1,:] @ c_L
            # Conduction Heat Flow in the Electronic System
            Flow_0E = self.elec_K (phi0_E, phi0_L)
            Flow_1E = self.elec_QE(phi0_E, phi0_L)
            Flow_2E = self.elec_QL(phi0_E, phi0_L)
            Flow_0E *= phi2_E
            Flow_1E *= phi1_E**2
            Flow_2E *= phi1_E*phi1_L
            # Conduction Heat Flow in the Lattice System
            Flow_0L = self.latt_K (phi0_E, phi0_L)
            Flow_1L = self.latt_QE(phi0_E, phi0_L)
            Flow_2L = self.latt_QL(phi0_E, phi0_L)
            Flow_0L *= phi2_L
            Flow_1L *= phi1_L*phi1_L
            Flow_2L *= phi1_L**2
            # Store
            self.debug[0,i] = Flow_0E[0] + Flow_1E[0] + Flow_2E[0]
            self.debug[1,i] = self.G(phi0_E[0], phi0_L[0])*(phi0_L-phi0_E)[0]
            # Diffusion Equation
            dphi_E = Flow_0E + Flow_1E + Flow_2E + self.G(phi0_E, phi0_L)*(phi0_L-phi0_E) + source[i,1:-1]
            dphi_L = Flow_0L + Flow_1L + Flow_2L + self.G(phi0_E, phi0_L)*(phi0_E-phi0_L)
            dphi_E /= self.elec_C(phi0_E, phi0_L)
            dphi_L /= self.latt_C(phi0_E, phi0_L)
            # Calculate the new value of the Temperature
            phi0_E += self.time_step * dphi_E
            phi0_L += self.time_step * dphi_L
            # Compute New Coefficients
            c_E = self.I0 @ np.hstack([0,phi0_E,0])
            c_L = self.I0 @ np.hstack([0,phi0_L,0])
            # Store The Temperature on the refined grid in a variable
            phi_E[i] = self.P0 @ c_E; phi_L[i] = self.P0 @ c_L
        # END OF THE MAIN LOOP
        end_EL = time.time()
        self.warning(0, str(end_EL - start_EL))
        return self.y, self.t, np.rollaxis(np.dstack([phi_E, phi_L]),2)


# =============================================================================
#
# -----------------------------------------------------------------------------
    def stability(self, LSM):
        # Useful Constant
        test  = np.linspace(270, self.burn, 50)
        # Worst case for the Diffusion Instability
        DE = max(self.elec_K(test, test)/self.elec_C(test, test))
        DL = max(self.latt_K(test, test)/self.latt_C(test, test))
        # Worst case for the Coupling Instability
        XE = max(self.G(test, test)/self.elec_C(test, test))
        XL = max(self.G(test, test)/self.latt_C(test, test))
        # Instability due to Diffusion
        DIF  = np.kron(np.diag(np.array([DE,DL])), LSM)
        # Instability due to Coupling
        EXC = np.kron(np.array([[-XE, XE],[XL, -XL]]), np.eye(len(LSM)))
        # Total Instability
        StbMat = EXC + DIF
        # Evaluate the Eigenvalues
        eigs = min(np.real(np.linalg.eig(StbMat)[0]))
        eigs = min(eigs, -XE/.5, -XL/.5)
        # Return the smallest time step
        return -1.9/eigs

# =============================================================================
#
# =============================================================================
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
            ' The manually chosen time step of ' + arg1 + ' very big and may cause instabilities in the simulation.\n' + \
            ' We suggest a timestep of ' + arg2 + ' s\n'
        if msg == 3:
            text = \
            ' The maunually chosen time step of ' + arg1 + ' is very small and may slow down the simulation.\n' + \
            ' We suggest a timestep of ' + arg2 + ' s\n'
        if msg == 4:
            text = \
            ' The source is not set, please recontrol the code. The command is sim.setsource(source)\n'
        print(\
    '--------------------------------------------------------------------------------------------------------------\n' + \
    text + \
    '--------------------------------------------------------------------------------------------------------------\n')
