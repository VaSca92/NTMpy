import sys
sys.path.insert(0, './code/')

from Sim2T import Sim2T # type: ignore
import Visual as vs # type: ignore
from Source import source # type: ignore
import numpy as np
import matplotlib.pyplot as plt


case = 2

if case == 1:

    s = source() # default option, Gaussian pulse
    s.setLaser(1, 8e-12)
    s.delay       = 2e-12       # time the maximum intensity hits
    s.refraction  = [1,1]
    s.absorption  = [1.9e-7, 1.9e-7]


    # initialize simulation: ntm.simulation(number of temperatures, source)
    sim = Sim2T() # Default initial temperature: 300 Kelvin
    sim.setSource(s)

    # add material layers:
    sim.addLayer( 50e-9, [10, 0], [lambda T: .112*T, 450], 6400, 6e17, 9)
    sim.addLayer( 50e-9, [10, 0], [lambda T: .112*T, 450], 6400, 6e17, 9)

    # set final simulation time (in seconds)
    sim.final_time = 50e-12

    # Run simulation
    [x, t, phi] = sim.run()

    vs.average(x,t,phi)
    

elif case == 2:
    # Case 2 ==================================================================
    # Initialize source
    s = source() # default option, Gaussian pulse
    s.setLaser(5, 2e-12)
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
    [x, t, phi] = sim.run()
    
    # Plot temperature
    vs.compare(x,t,phi[0],phi[1])

    # -------------------------------------------------------------------------

elif case == 3:
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
    [x, t, T] = sim.run()
    t = np.arange(sim.start_time, sim.final_time, sim.time_step)
    T_e = T[0]; T_l = T[1]
    
    vs.average(x,t,T)

    exp_weights = np.exp(-x/1e-8)
    avT_E = np.average(T_e,axis = 1, weights = exp_weights)
    avT_L = np.average(T_l,axis = 1, weights = exp_weights)
    avT_tot = (avT_E + avT_L - 600)

    plt.figure()
    plt.plot(t*1e12, avT_tot)
    plt.grid()
