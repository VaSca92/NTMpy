from Sim2T import Sim2T
import Visual as vs
from Source import source
import numpy as np
import matplotlib.pyplot as plt


case = 3

if case == 1:
    # Case 1 ==================================================================
    # Create Source
    s = source()
    s.absorption = [1e-8, 1e-8]
    s.refraction = [1, 1]
    s.time = 2e-12
    s.delay = 2e-12
    s.angle = np.pi/4
    # Instantiate the Sim2T class
    sim = Sim2T()
    # Add layers (Length,conductivity,heatCapacity,rho,coupling)
    sim.addLayer(40e-9, [ 6,  1], [lambda T: .112*T, 450], 6500, 5e17, 12)
    sim.addLayer(80e-9, [12,  1], [lambda T: .025*T, 730], 5100, 5e17, 12)

    sim.setSource(s)

    sim.final_time = 10e-12

    #[phi_E,phi_L,x] = sim.run()
    [phi,x] = sim.run()

    t = np.arange(sim.start_time, sim.final_time, sim.time_step)
    #vs.compare(x,t,phi[0],phi[1], 2)

    vs.spaceVStime(x, t, phi[0])
    #vs.surface(x,t,s.matrix(x,t,sim.length))

    # -------------------------------------------------------------------------

elif case == 2:
    # Case 2 ==================================================================

    L  = 5e-6      # Length of the Grating
    Ce = 2e+4      # Specific Heat Electrons
    Cl = 2.5e6     # Specific Heat Lattice
    ke = 3.2e+1    # Conductivity Electrons
    kl = 2.75      # Conductivity Lattica
    G  = 3e+16     #Exchange constant
    # Define the Object
    sim = Sim2T()
    # Define Media
    sim.addLayer(L, [ke,kl], [Ce, Cl], 1, G)
    # Define Initial Condition
    sim.init_E = lambda x: np.sin(2*np.pi*x/L)
    sim.init_L = 0
    # Assign Boundary Condition type
    sim.LBCT_E = 0
    sim.RBCT_E = 0
    sim.LBCT_L = 0
    sim.RBCT_L = 0
    sim.LBCV_E = 0
    sim.RBCV_E = 0
    sim.LBCV_L = 0
    sim.RBCV_L = 0
    # Chose Final  Time
    sim.final_time = 8000

    # RUN ...
    [phi, x] = sim.run()
    t = np.arange(sim.start_time, sim.final_time, sim.time_step)

    #vs.compare(x,t,phi[0],phi[1],1)
    #plt.plot(x, phi[0][1,:], x, phi[1][1,:], x, phi[0][-1,:], x, phi[1][-1,:])
    #plt.grid();plt.show()

    q = 2*np.pi/L
    # Find Analitical Values of gamma1 and gamma2
    A = 1; B = (ke/Ce+kl/Cl)*q**2 + G*(1/Ce+1/Cl)
    C = G*(ke+kl)*q**2/(Ce*Cl) + q**4*(ke*kl)/(Ce*Cl)
    # Resolutive formula for 2nd grade equations
    delta = B**2-4*A*C

    index = np.where(phi[0][0,:] == np.max(phi[0][0,:]))[0]

    dt = sim.time_step
    g2 = -(phi[0][-2, index]-phi[0][-1, index])/(dt*phi[0][-1, index])
    g3 = -np.log(phi[0][-2, index]/phi[0][-1, index])/dt
    g1 = - (-B+np.sqrt(delta))/2


    print("Analytic value of the time constant:    " + str(1/g1))
    print("Numeric (1) value of the time constant: " + str(-1/g2[0]))
    print("Numeric (2) value of the time constant  " + str(-1/g3[0]))

    # -------------------------------------------------------------------------

elif case == 3:
    # Case 3 ==================================================================
    # Initialize source
    s = source() # default option, Gaussian pulse
    s.setLaser(1, 2e-12)
    s.delay       = 2e-12       # time the maximum intensity hits
    s.refraction  = [1,1]
    s.absorption  = [1.9e-7, 1.9e-7]


    # initialize simulation: ntm.simulation(number of temperatures, source)
    sim = Sim2T() # Default initial temperature: 300 Kelvin
    sim.setSource(s)

    # add material layers:
    sim.addLayer( 30e-9, [ 8,  0], [lambda T: .112*T, 450], 6500, 6e17, 9)
    sim.addLayer( 80e-9, [24, 12], [lambda T: .025*T, 730], 5100, 6e17, 12)

    # set final simulation time (in seconds)
    sim.final_time = 50e-12

    # Run simulation
    [phi, x] = sim.run()

    t = np.arange(sim.start_time, sim.final_time, sim.time_step)
    vs.average(x,t,phi)

    # -------------------------------------------------------------------------

elif case == 4:
    # Setup source
    s = source()
    s.setLaser(60, .2e-12)
    s.delay = 1e-12
    s.angle = np.pi/4
    s.polarization  = 'p'
    #s.type_x = "lb"

    # Platinum properties
    length_Pt   = 10e-9
    n_Pt        = 1.7176
    k_el_Pt     = 72
    rho_Pt      = 1e3*21
    C_el_Pt     = lambda Te: 740/(1e3*21)*Te
    C_lat_Pt    = 2.78e6/rho_Pt
    G_Pt        = 2.5e17

    # Silicon properties
    n_Si        = 5.5674
    k_el_Si     = 130
    k_lat_Si    = lambda T: np.piecewise(T,[T<=120.7,T>120.7],\
                                      [lambda T: 100*(0.09*T**3*(0.016*np.exp(-0.05*T)+np.exp(-0.14*T))),
                                       lambda T: 100*(13*1e3*T**(-1.6))])
    rho_Si      = 2.32e3
    C_el_Si     = lambda Te: 150/rho_Si*Te
    C_lat_Si    = 1.6e6/rho_Si
    G_Si        = 18e17

    #
    s.refraction = [n_Pt, n_Si, n_Si, n_Si]
    s.absorption = [1e-8, 1e-8, 1e-8, 1e-8]
    sim = Sim2T()
    sim.setSource(s)

    sim.addLayer(length_Pt,[k_el_Pt,k_el_Pt],[C_el_Pt,C_lat_Pt],rho_Pt,[G_Pt],12)
    sim.addLayer(100e-9,[k_el_Si,k_lat_Si],[C_el_Si,C_lat_Si],rho_Si,[G_Si],15)
    sim.addLayer(400e-9,[k_el_Si,k_lat_Si],[C_el_Si,C_lat_Si],rho_Si,[G_Si],15)
    sim.addLayer(1600e-9,[k_el_Si,k_lat_Si],[C_el_Si,C_lat_Si],rho_Si,[G_Si],15)

    sim.final_time = 6e-12
    [T,x] = sim.run()
    t = np.arange(sim.start_time, sim.final_time, sim.time_step)
    T_e = T[0]; T_l = T[1]
    
    vs.surface(x,t,s.matrix(x,t,sim.length))
    vs.average(x,t,T)

    exp_weights = np.exp(-x/1e-8)
    avT_E = np.average(T_e,axis = 1, weights = exp_weights)
    avT_L = np.average(T_l,axis = 1, weights = exp_weights)
    avT_tot = (avT_E + avT_L - 600)

    plt.figure()
    plt.plot(t*1e12, avT_tot)
    plt.grid()

