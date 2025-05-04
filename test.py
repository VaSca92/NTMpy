from NTMpy import NTMpy as ntm


# Initialize source
s = ntm.source() # default option, Gaussian pulse
s.fluence = 1  # Energy per meter square
s.FWHM    = 2e-12 # Full Width Half Maximum (duration of the pulse)
s.t0      = 2e-12 # time the maximum intensity hits
s.lambda_vac  = 400 # Wavelength of source (in nanometers)

# initialize simulation: ntm.simulation(number of temperatures, source)
sim = ntm.simulation(2,s) # Default initial temperature: 300 Kelvin


# add material layers:
# > non constant quantities are inserted as lambda functions
# > if more temperature are used, conductivity and heat capacity are vectors
# addlayer  ( Length, refractive index, conductivity, Heat Capacity, density, coupling)
sim.addLayer( 30e-9, 1+3j, [ 8,  0], [lambda T: .112*T, 450], 6500, 6e17) # first layer
sim.addLayer( 80e-9, 1+3j, [24, 12], [lambda T: .025*T, 730], 5100, 6e17) # second layer

# set final simulation time (in seconds)
sim.final_time = 50e-12

# Run simulation
[x,t,phi] = sim.run()

# Link the plotting library to the simulation
vs = ntm.visual(sim)
# Show animation (like the one in the github page)
vs.animation(1) # input is the animation speed
print(phi)
