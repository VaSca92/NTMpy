# =============================================================================
#   TITLE: NTMpy sim2T
# -----------------------------------------------------------------------------
#        Authors: Valentino Scalera, Lukas Alber
#        Version: 1.0
#   Dependencies: numpy, matplotlib, bspline
# -----------------------------------------------------------------------------

import numpy as np
from bspline import Bspline
from bspline.splinelab import aptknt
import time

from Visual import *
from Source import source

# =============================================================================
#
# =============================================================================
class simulation(object):

# =============================================================================
#
# -----------------------------------------------------------------------------
    def __init__(self):
        # Time Data
        self.start_time  = 0
        self.final_time  = 0
        self.time_step   = 0
        self.dt_ext      = []
        # Geometric data
        self.x          = np.array([])
        self.y          = np.array([])
        self.length     = [0]
        self.grd_points = [ ]
        self.plt_points = [ ]
        self.order      = 5
        # Electronic System
        self.elec_K  = []
        self.elec_Q  = []
        self.elec_C  = []
        self.LBCT_E  = 1
        self.RBCT_E  = 1
        self.LBCV_E  = 0
        self.RBCV_E  = 0
        self.init_E  = 300
        # Lattice System
        self.latt_K  = []
        self.latt_Q  = []
        self.latt_C  = []
        self.LBCT_L  = 1
        self.RBCT_L  = 1
        self.LBCV_L  = 0
        self.RBCV_L  = 0
        self.init_L  = 300
        # Differentiation Matrices an Plot Matrix
        self.D0 = np.zeros([0,0])
        self.D1 = np.zeros([0,0])
        self.D2 = np.zeros([0,0])
        self.P0 = np.zeros([0,0])
        # Coupling
        self.G = []
        # Indices of the equation type for the Electronic System
        self.diffusionE = []
        self.heat_flowE = []
        self.continousE = []
        # Indices of the equation type for the Lattice System
        self.diffusionL = []
        self.heat_flowL = []
        self.continousL = []
        # Incides of the interfaces and number of Layers
        self.layers     = 0
        self.interface = []
        # Conductivity == 0
        self.zeroE = [True]
        self.zeroL = [True]
        # Source
        self.source = source()
        self.source.peak = 0
        # Default Settings
        self.default_Ng = 12
        self.default_Np = 60

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
    def addLayer(self, L, K, C, D, G = 0, Ng = False, Np = False):
        # Add Layer to the Electron system # # # # # # # # # # # # # # # # # #
        # Thermal Conductivity
        self.elec_K.append(self.lambdize(K[0]))
        # Heat Capacity  (specific heat * density)
        self.elec_C.append(self.lambdize(C[0], D))
        # Derivative of the thermal conductivity
        dummyQE = self.elec_K[-1]
        self.elec_Q.append(lambda x: (dummyQE(x+1e-9)-dummyQE(x))/1e-9)
        # Add Layer to the Lattice system  # # # # # # # # # # # # # # # # # #
        # Thermal Conductivity
        self.latt_K.append(self.lambdize(K[-1]))
        # Heat Capacity  (specific heat * density)
        self.latt_C.append(self.lambdize(C[-1], D))
        # Derivative of the thermal conductivity
        dummyQL = self.latt_K[-1]
        self.latt_Q.append(lambda x: (dummyQL(x+1e-9)-dummyQL(x))/1e-9)
        # Add Coupling # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.G = np.append(self.G, G)
        # Detect Zeros
        self.zeroE.append(K[ 0] == 0)
        self.zeroL.append(K[-1] == 0)
        # Add Geometry
        self.length.append(L + self.length[-1])
        Ng = self.default_Ng if not Ng else Ng
        self.grd_points.append(Ng)
        Np = self.default_Np if not Np else Np
        self.plt_points.append(Np)


# =============================================================================
#
# -----------------------------------------------------------------------------
    def setSource(self, source, n = 1):
        self.source = source


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
                dummy = fun(0,0)
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
        interface  = [-1]
        # Select Order of the Spline (future expert setting)
        order = self.order
        # Layer Stability Matrix
        LSM = []
        # FOR each layer: Construct the Ai Matrices
        for i in range( len(length) - 1):
            # Assign the Equation type
            current = range( len(self.x), len(self.x) + grd_points[i] )
            self.detect_id(i, list(current))
            # Update Indices of the elements
            interface.extend([interface[-1] + 1, interface[-1] + grd_points[i]])
            # Define Space Points
            x = np.linspace(length[i], length[i+1] , grd_points[i])
            y = np.linspace(length[i], length[i+1] , plt_points[i])
            # Spline Generation
            knot_vector = aptknt( x, order)
            basis       = Bspline(knot_vector, order)
            # Generate Differentiation and Plot Matrices for the Layer
            D0L = basis.collmat(x, deriv_order = 0)
            D1L = basis.collmat(x, deriv_order = 1)
            D2L = basis.collmat(x, deriv_order = 2)
            P0L = basis.collmat(y, deriv_order = 0)
            # Correct BSpline Package Bug
            D0L[-1,-1] = 1; P0L[-1,-1] = 1; D1L[-1] = -np.flip(D1L[0],0)
            # Matrices for the Stablity
            LSM.append(np.array(np.dot(D2L,np.linalg.inv(D0L))))
            # Zero Matrices for the Concatenation of Diagonal Block Matrices
            ZHD = np.zeros([len(self.D0), len(D0L)])
            ZVD = np.zeros([len(D0L), len(self.D0)])
            ZHP = np.zeros([len(self.P0), len(D0L)])
            ZVP = np.zeros([len(P0L), len(self.D0)])
            # Total Differentiation and Plot Matrices
            self.D0 = np.block([[self.D0, ZHD],[ZVD,D0L]])
            self.D1 = np.block([[self.D1, ZHD],[ZVD,D1L]])
            self.D2 = np.block([[self.D2, ZHD],[ZVD,D2L]])
            self.P0 = np.block([[self.P0, ZHP],[ZVP,P0L]])
            # Extend the Space Grid
            self.x = np.append( self.x, x)
            self.y = np.append( self.y, y)
        self.interface = interface[2:-1]
        self.layers    = round(len(self.interface)*0.5)
        # Remove Equation dedicated to Boundary Condition
        L = len(self.x) - 1
        if 0 in self.heat_flowE: self.heat_flowE = self.heat_flowE[+1:]
        if 0 in self.heat_flowL: self.heat_flowL = self.heat_flowL[+1:]
        if L in self.heat_flowE: self.heat_flowE = self.heat_flowE[:-1]
        if L in self.heat_flowL: self.heat_flowL = self.heat_flowL[:-1]
        # Calculate approximated Time steps
        self.dt_ext = self.stability(LSM)


# =============================================================================
#
# -----------------------------------------------------------------------------
    def detect_id(self, i, indices):
        # Set indices for the Electronic System
        if self.zeroE[i+1]:
            self.diffusionE.append(indices[0:-1])
        elif not(self.zeroE[i] or self.zeroE[i+1]):
            self.continousE.append(indices[0])
            self.diffusionE.append(indices[1:-1])
            self.heat_flowE.append(indices[-1])
        elif self.zeroE[i] and not self.zeroE[i+1]:
            self.heat_flowE.append(indices[ 0])
            self.diffusionE.append(indices[1:-1])
            self.heat_flowE.append(indices[-1])
        # Set indices for the Lattice System
        if self.zeroL[i+1]:
            self.diffusionL.append(indices[0:-1])
        elif not(self.zeroL[i] or self.zeroL[i+1]):
            self.continousL.append(indices[ 0])
            self.diffusionL.append(indices[1:-2])
            self.heat_flowL.append(indices[-1])
        elif self.zeroL[i] and not self.zeroL[i+1]:
            self.heat_flowL.append(indices[ 0])
            self.diffusionL.append(indices[1:-2])
            self.heat_flowL.append(indices[-1])


# =============================================================================
#
# -----------------------------------------------------------------------------
    def generate_BC(self):
        # Constant for Type Checking
        typecheck = type(lambda x: x)
        # Initialization
        BC_E = np.zeros([2 ,len(self.t)]); BC_L = np.zeros([2 ,len(self.t)])
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
        # Return Boundary Condition
        return BC_E, BC_L

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
    def generate_matrix(self):
        # Matrics of Coefficient fot the Electron and Lattice System
        LHSE = self.D0.copy(); LHSL = self.D0.copy()
        # Setting Boundary Condition Type
        if self.LBCT_E == 1 and not self.zeroE[ 1]:
            LHSE[ 0] = -self.D1[ 0].copy()
        if self.RBCT_E == 1 and not self.zeroE[-1]:
            LHSE[-1] =  self.D1[-1].copy()
        if self.LBCT_L == 1 and not self.zeroL[ 1]:
            LHSL[ 0] = -self.D1[ 0].copy()
        if self.RBCT_L == 1 and not self.zeroL[-1]:
            LHSL[-1] =  self.D1[-1].copy()
        for k in self.continousE: LHSE[ k, k-1] = 1; LHSE[ k, k] = -1
        for k in self.continousL: LHSL[ k, k-1] = 1; LHSL[ k, k] = -1
        # Create the matrix for the Interface Condition
        HF = self.D1[self.interface]
        return LHSE, LHSL, HF

# =============================================================================
#
# -----------------------------------------------------------------------------
    def run(self):
        # ---------------------------------------------------------------------
        # Setup Phase: Timestep evaluation, Source Generation,
        #              Boundary Condition prepared, Adjust Coupling
        # ---------------------------------------------------------------------
        self.build_geometry()
        # Load all the Geometry Matrices --------------------------------------
        # Rename some Instance Variables
        Ng = self.grd_points
        BCLL = 1 - self.zeroL[1]; BCLR = 1 - self.zeroL[-1]
        BCEL = 1 - self.zeroE[1]; BCER = 1 - self.zeroE[-1]
        # STABILITY EVALUATION ################################################
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
        # Generate all the matrices ###########################################
        LHSE, LHSL, HF     = self.generate_matrix()
        BC_E, BC_L         = self.generate_BC()
        c_E, u_E, c_L, u_L = self.generate_init()
        # SOURCE GENERATION ###################################################
        source = self.source.matrix(self.x, self.t, self.length)
        # ------------------------------------------- Setup ended -------------

        # ---------------------------------------------------------------------
        #  MAIN LOOP
        # ---------------------------------------------------------------------
        # Rename Boundary Condition type
        LBCE = self.LBCT_E; RBCE = self.RBCT_E;
        LBCL = self.LBCT_L; RBCL = self.RBCT_L;
        # Initialization of the variables for the Electronic System
        phi_E    = np.zeros([len(self.t),len(self.y)])
        Flow_1E  = np.zeros( len(self.x) )
        Flow_2E  = np.zeros( len(self.x) )
        dphi_E   = np.zeros( len(self.x) )
        RHSE     = np.zeros( len(self.x) )
        #Initialization of the variables for the Lattice System
        phi_L    = np.zeros([len(self.t),len(self.y)])
        Flow_1L  = np.zeros( len(self.x) )
        Flow_2L  = np.zeros( len(self.x) )
        dphi_L   = np.zeros( len(self.x) )
        RHSL     = np.zeros( len(self.x) )
        # Set Initial Condition
        phi_E[0] = u_E; phi_L[0] = u_L
        # Identify Layer
        P = np.cumsum([np.append([0], Ng[:-1]), Ng], axis=1).transpose().astype(int)
        # HERE STARTS THE MAIN LOOP
        start_EL = time.time()
        for i in range(1,len(self.t)):
            # Go from coefficient c to phi and its derivatives
            phi0_E = np.dot(self.D0,c_E); phi0_L = np.dot(self.D0,c_L)
            phi1_E = np.dot(self.D1,c_E); phi1_L = np.dot(self.D1,c_L)
            phi2_E = np.dot(self.D2,c_E); phi2_L = np.dot(self.D2,c_L)
            for j in range(len(P)):  # For every Layer
                # Conduction Heat Flow in the Electronic System
                Flow_1E[P[j,0]:P[j,1]] = self.elec_Q[j](phi0_E[P[j,0]:P[j,1]])
                Flow_2E[P[j,0]:P[j,1]] = self.elec_K[j](phi0_E[P[j,0]:P[j,1]])
                Flow_1E[P[j,0]:P[j,1]] *= phi1_E[P[j,0]:P[j,1]]**2
                Flow_2E[P[j,0]:P[j,1]] *= phi2_E[P[j,0]:P[j,1]]
                # Conduction Heat Flow in the Lattice System
                Flow_1L[P[j,0]:P[j,1]] = self.latt_Q[j](phi0_L[P[j,0]:P[j,1]])
                Flow_2L[P[j,0]:P[j,1]] = self.latt_K[j](phi0_L[P[j,0]:P[j,1]])
                Flow_1L[P[j,0]:P[j,1]] *= phi1_L[P[j,0]:P[j,1]]**2
                Flow_2L[P[j,0]:P[j,1]] *= phi2_L[P[j,0]:P[j,1]]
                # Diffusion Equation
                dphi_E[P[j,0]:P[j,1]] = Flow_1E[P[j,0]:P[j,1]] + Flow_2E[P[j,0]:P[j,1]] + self.G[j]*(phi0_L[P[j,0]:P[j,1]]-phi0_E[P[j,0]:P[j,1]]) + source[i,P[j,0]:P[j,1]]
                dphi_L[P[j,0]:P[j,1]] = Flow_1L[P[j,0]:P[j,1]] + Flow_2L[P[j,0]:P[j,1]] + self.G[j]*(phi0_E[P[j,0]:P[j,1]]-phi0_L[P[j,0]:P[j,1]])
                dphi_E[P[j,0]:P[j,1]] /= self.elec_C[j](phi0_E)[P[j,0]:P[j,1]]
                dphi_L[P[j,0]:P[j,1]] /= self.latt_C[j](phi0_L)[P[j,0]:P[j,1]]
            # Apply Heat Conservation on surfaces
            for k, j in enumerate(self.heat_flowE): # For every interface
                # Calculate the Flux into and out from the interface
                IFconL = self.elec_K[ k ](phi0_E[j])*HF[ 2*k ]
                IFconR = self.elec_K[k+1](phi0_E[j])*HF[2*k+1]
                # Electronic System
                LHSE[j] = -IFconL
                LHSE[j] += IFconR
            for k, j in enumerate(self.heat_flowL):
                # Calculate the Flux into and out from the interface
                IFconL = self.latt_K[ k ](phi0_L[j])*HF[ 2*k ]
                IFconR = self.latt_K[k+1](phi0_L[j])*HF[2*k+1]
                # Lattice System
                LHSL[j]  = -IFconL
                LHSL[j]  += IFconR
            # Applying Explicit Euler Method
            RHSE = phi0_E + self.time_step * dphi_E
            RHSL = phi0_L + self.time_step * dphi_L
            # Make Room for Boundary Condition and Interface Condition
            if BCEL: RHSE[ 0] = BC_E[0,i]/self.elec_K[ 0](phi0_E[ 0])**LBCE
            if BCER: RHSE[-1] = BC_E[1,i]/self.elec_K[-1](phi0_E[-1])**RBCE
            if BCLL: RHSL[ 0] = BC_L[0,i]/self.latt_K[ 0](phi0_L[ 0])**LBCL
            if BCLR: RHSL[-1] = BC_L[1,i]/self.latt_K[-1](phi0_L[-1])**RBCL
            RHSE[self.heat_flowE] = 0; RHSE[self.continousE] = 0
            RHSL[self.heat_flowL] = 0; RHSL[self.continousL] = 0
            # Calculate the new value of the Temperature
            c_E = np.linalg.solve(LHSE,RHSE)
            c_L = np.linalg.solve(LHSL,RHSL)
            # Store The Temperature on the refined grid in a variable
            phi_E[i] = np.dot(self.P0,c_E); phi_L[i] = np.dot(self.P0,c_L)
        # END OF THE MAIN LOOP
        end_EL = time.time()
        self.warning(0, str(end_EL - start_EL))
        return np.rollaxis(np.dstack([phi_E, phi_L]),2), self.y


# =============================================================================
#
# -----------------------------------------------------------------------------
    def stability(self, LSM):
        # Useful Constant
        test       = np.linspace(270,2000,50)
        eigs       = np.zeros([len(LSM)])
        for i in range(self.layers + 1):
            dim = len(LSM[i]); Z = np.zeros([dim,dim])
            # Worst case for the Diffusion Instability
            DE = max(self.elec_K[i](test)/self.elec_C[i](test))
            DL = max(self.latt_K[i](test)/self.latt_C[i](test))
            # Worst case for the Coupling Instability
            XE = max(self.G[i]/self.elec_C[i](test))
            XL = max(self.G[i]/self.latt_C[i](test))
            # Instability due to Diffusion
            DIF  = np.block([[DE*LSM[i], Z],[Z, DL*LSM[i]]])
            # Instability due to Coupling
            EXC = np.kron(np.array([[-XE, XE],[XL, -XL]]), np.eye(dim))
            # Total Instability
            StbMat = EXC + DIF
            # Evaluate the Eigenvalues
            eigs[i] = min(np.real(np.linalg.eig(StbMat)[0]))
            eigs[i] = min(eigs[i], -XE/.3, -XL/.3)
            #print("Line 446-448")
            #print(-1.9/min(np.real(np.linalg.eig(StbMat)[0])))
            #print(-1.9/(-XE/.3))
        # Return the smallest time step
        return min(-1.9/eigs)

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
            ' The manually chosen time step of ' + arg1 + ' is eventually too big and could cause instabilities in the simulation.\n' + \
            ' We suggest a timestep of ' + arg2 + ' s\n'
        if msg == 3:
            text = \
            ' The maunually chosen time step of ' + arg1 + ' is very small and will eventually cause a long simulation time.\n' + \
            ' We suggest a timestep of' + arg2 + ' s\n'
        print(\
    '--------------------------------------------------------------------------------------------------------------\n' + \
    text + \
    '--------------------------------------------------------------------------------------------------------------\n')
