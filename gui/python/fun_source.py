import eel
import numpy as np

from init_variables import flags, laser, layers, nindex, src
from main import src_init

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
    laser["energy"] = int(energy)
    laser["fwhm"]   = int(fwhm)
    laser["delay"]  = int(delay)

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
    return src.create(np.linspace(0,total_leng,128), np.array([src.delay]))

@eel.expose
def plot_src_t():
    src_init()
    total_time = 4*laser["fwhm"] + laser["delay"]
    return src.create( np.array([0]), np.linspace(0, total_time, 128))
   

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