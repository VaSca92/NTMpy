import eel
# Set web files folder and optionally specify which file types to check for eel.expose()

flags = {"reflection": False, "source_set": False}
laser = {"energy": 0, "fwhm": 0, "delay": 0}
layers = []
nindex = []

# Material properties interfaces #################
@eel.expose
def setLayers(layer, id = -1):
    layers.append(layer)
    nindex.append({"l": "", "nr": "", "nr": ""})

@eel.expose
def getLayers():
    return layers 

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

# Absorption / Refraction interface ##############
@eel.expose
def setIndexN( n, id):
    if flags["reflection"]:
        nindex[id]["nr"] = n[0]
        nindex[id]["nr"] = n[1]
    else:
        nindex[id]["l"] = n


@eel.expose
def getIndexN():
    return nindex

eel.init('web', allowed_extensions=['.js', '.html', '.css'])
eel.start('html/main.html', size=(1080, 680), jinja_templates='html')    # Start

