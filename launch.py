import sys
sys.path.insert(0, './code')
sys.path.insert(0, './gui/python')

import eel
from Sim2T import Sim2T # type: ignore
from Source import source # type: ignore

import matplotlib.pyplot as plt
import numpy as np


# Set web files folder and optionally specify which file types to check for eel.expose()

flags = {"reflection": False, "source_set": False}
laser = {"energy": 0, "fwhm": 0, "delay": 0}
layers = []
nindex = []

src = source()
data = []

# Material properties interfaces #################
@eel.expose
def setLayers(layer, id = -1):

    if id < 0:
        layers.append(layer)
        nindex.append({"l": "", "nr": "", "ni": ""})
    else:
        layers[id] = layer
    

@eel.expose
def getLayers():
    return layers 

# Modify layers order 
@eel.expose
def move_layer(id1, id2):
    layers[id1], layers[id2] = layers[id2], layers[id1]
    nindex[id1], nindex[id2] = nindex[id2], nindex[id1]

@eel.expose
def remove_layer(id):
    layers.pop(id)
    nindex.pop(id)

@eel.expose
def duplicate_layer():
    pass


# Flags interfaces ###############################
@eel.expose
def setFlags(id, prop): 
    flags[id] = prop

@eel.expose
def getFlags(prop):
    return flags[prop]

# Laser source interface #########################
@eel.expose
def setSource(energy, fwhm, delay):
    flags["source_set"] = True
    laser["energy"] = energy
    laser["fwhm"]   = fwhm
    laser["delay"]  = delay

@eel.expose
def getSource():
    return laser

@eel.expose
def source():
    pass

@eel.expose
def plot_src_x():
    src_init()
    total_leng = np.sum([ layer["length"] for layer in layers])
    data["src"] = src.create(np.linspace(0,total_leng,128), np.array([src.delay]))
    return data

@eel.expose
def plot_src_t():
    src_init()
    total_time = 4*laser["fwhm"] + laser["delay"]
    data["src"] = src.create( np.array([0]), np.linspace(0, total_time, 128))
    return data    

# Absorption / Refraction interface ##############
@eel.expose
def setIndexN( nr, ni, id):
    if flags["reflection"]:
        nindex[id-1]["nr"] = nr
        nindex[id-1]["ni"] = ni
    else:
        nindex[id-1]["l"] = nr

@eel.expose
def getIndexN():
    return nindex


# Save all #######################################
@eel.expose
def save():
     pass

# Run Simulation #################################
@eel.expose
def run():
    sim = Sim2T()
    for layer in layers:
        length = float(layer["length"])
        cond = [eval("lambda Te, Tl: " + layer["K"][0]), eval("lambda Te, Tl: " + layer["K"][1])]
        capc = [eval("lambda Te, Tl: " + layer["C"][0]), eval("lambda Te, Tl: " + layer["C"][1])]
        coup =  eval("lambda Te, Tl: " + layer["G"])
        dens = float(layer["rho"])
        sim.addLayer( length, cond, capc, dens, coup)


# Util and Functions
def src_init():
    if not flags["source_set"]:
        src.angle = 0
        src.setLaser(float(laser["energy"]), float(laser["fwhm"]))
        src.absorption = [float(n["l"]) for n in nindex]
        src.delay = float(laser["delay"])
        #total_leng = np.sum([ layer["length"] for layer in layers])
        #total_time = 4*laser["fwhm"] + laser["delay"]
        #srcplt = src.create(np.linspace(0,total_leng,128), np.linspace(0, total_time, 128))
        #flags["source_set"] = True


eel.init('.', allowed_extensions=['.js', '.html', '.css'])
eel.start('gui/html/main.html', size=(1120, 720), jinja_templates='gui/html')    # Start

