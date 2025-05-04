# What does NTMpy solves
NTMpy is a python package to solve 1D diffusion equation with 2 or 3 temperature coupled

The equation under consideration is:

$$
\begin{cases}
C_{Ei}\cdot\rho_i\cdot\partial_t T_E = \partial_x(k_{Ei}\cdot\partial_xT_E)+G_{ELi}\cdot(T_L-T_E)+G_{ESi}\cdot(T_S-T_E)+S(x,t)\\
C_{Li}\cdot\rho_i\cdot\partial_t T_L = \partial_x(k_{Li}\cdot\partial_xT_L)+G_{ELi}\cdot(T_E-T_L)+G_{LSi}\cdot(T_S-T_L)\\
C_{Si}\cdot\rho_i\cdot\partial_t T_S = \partial_x(k_{Si}\cdot\partial_xT_S)+G_{ESi}\cdot(T_E-T_S)+G_{LSi}\cdot(T_L-T_S)
\end{cases}
$$
 
where </br>
$T_E(x,t)$, $T_L(x,t)$, and $T_S(x,t)$ are temperature (subscripts recall Electron, Lattice, and Spin) </br>
$\rho$ is the mass density in the $i$-th layer of material (the material properties are piecewise homogeneous) </br>
$C_{Ki}$ and $k_{Ki}$ are the specific heat and thermal conductivity of the system $K = E,L,S$ in the $i$-th layer of material </br>
$G_{KHi}$ is the thermal coupling coefficient between the system $K$ and $H$ (with $K,H\in \lbrace E,L,S\rbrace$) in the $i$-th layer </br>
$S(x,t)$ is the energy generation/ source </br>

The material properties $C_{Ki}(T_E,T_L,T_S)$, $k_{Ki}(T_E,T_L,T_S)$, and $G_{KH}(T_E,T_L,T_S)$ are function of the three temperatures </br>
The dependance on temperature are non-dispersive, i.e. they depend on the temperature in the same point at the same time

$$C_{Ki}(x,t) = C_{Ki}(T_E(x,t),T_L(x,t),T_S(x,t))\ ,\qquad k_{Ki}(x,t) = k_{Ki}(T_E(x,t),T_L(x,t),T_S(x,t))\ ,\qquad G_{KHi}(x,t) = G_{KHi}(T_E(x,t),T_L(x,t),T_S(x,t))$$

The solution is obtained using the Finite Element Method (FEM) + explicit Euler method. The FEM uses B-splines + collocation method. 

Further informations can be found on the [paper](https://www.sciencedirect.com/science/article/pii/S0010465521001028?via%3Dihub).

------------------------------------------------------------------------------------------------------------------

# How to install
The standard version of NTMpy, which follows the documentation on the original udcm-su [repository](https://github.com/udcm-su/NTMpy), can be installed via `pip install NTMpy`.</br>
The version on this repository is a successive version, currently under development. You can use it by cloning the repository and instering the code fonder in your path or by manually importing it.

------------------------------------------------------------------------------------------------------------------

# How to use NTMpy
NTMpy provides a set of classes to simulate coupled heat equation with 2 or 3 temperature in a multilayer system.
The simulation parameters are customizable, but the codes can choose them automatically if not given by the user.

Here is an example:
```python
from Sim2T import Sim2T 
import Visual as vs 
from Source import source 

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
# addLayer(length, [CE, CL], [kE, kL], density, G_EL, spline_number)
sim.addLayer( 30e-9, [ 8,  0], [lambda T: .112*T, 450], 6500, 6e17, 9)
sim.addLayer( 80e-9, [24, 12], [lambda T: .025*T, 730], 5100, 6e17, 12)

# set final simulation time (in seconds)
sim.final_time = 50e-12

# Run simulation
[x, t, phi] = sim.run()
    
# Plot temperature
vs.compare(x,t,phi[0],phi[1])
```

The ouput `phi` is a 3D array with the following structure:
* the first index selects the temperature: `phi[0]` is the electron temperature, `phi[1]` is the lattice temperature.
* the second index is the time instant: `phi[0][0]` is the initial temperature, `phi[0][100]` is the temperature after 100 time steps.
* the third index is the space position: `phi[0][100][10]` is the temperature of the temperature in the 10th point of the grid.

------------------------------------------------------------------------------------------------------------------

# How to contribute
New features for NTMpy are currently under development, but with a very slow pace. A graphic interface is under development using [eel](https://github.com/python-eel/Eel).</br>
If there is any important feature you think is missing and it is important for the experiments, you can open an issue on this repository.</br>
If you want to contribute to the code development, you can either clone this directory or contact the author [Valentino Scalera](mailto:valentino.scalera@uniparthenope.it).</br>
Any collaboration would be very appreciated

------------------------------------------------------------------------------------------------------------------

## About the authors 
NTMpy was a project of the [Ultrafast Condensed Matter physics groupe](http://udcm.fysik.su.se/) in Stockholm. The main contributors are: 
* [Lukas Alber](https://github.com/luksen99) 
* [Valentino Scalera](https://github.com/VaSca92) 
* [Vivek Unikandanunni](https://github.com/VivekUUnni)
* Stefano Bonetti

At the moment, only Valentino Scalera is handling the maintenance and development of NTMpy.</br>
You can contact him via [email](mailto:valentino.scalera@uniparthenope.it) if you need help or you want to contribute.

## Cite 
Please, cite the reference [paper](https://www.sciencedirect.com/science/article/pii/S0010465521001028):</br>
`@article{alber2020ntmpy,
    title={NTMpy: An open source package for solving coupled parabolic differential equations in the framework of the three-temperature model},
    author = {Lukas Alber and Valentino Scalera and Vivek Unikandanunni and Daniel Schick and Stefano Bonetti},
    journal = {Computer Physics Communications},
    year={2021},
    volume = {265},
    pages = {107990},
    issn = {0010-4655},
    doi = {https://doi.org/10.1016/j.cpc.2021.107990},
    url = {https://www.sciencedirect.com/science/article/pii/S0010465521001028}
}`

## Dependencies:
NTMpy has four dependencies: [Numpy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/), [B-splines](https://github.com/johntfoster), and [Progressbar](https://pypi.org/project/tqdm/2.2.3/)

Note that by downloading the package via `pip install NTMpy` a routine, which automatically checks if all the required packages are existent on the local machine is implemented. If one of the dependent pip packages, listed here, is missing an automatic download is initiated.

  

       
  

 
