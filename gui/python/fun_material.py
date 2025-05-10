import eel
from init_variables import layers, nindex


# Material properties interfaces #################
@eel.expose
def setLayers(layer, id = -1):

    if id < 0:
        layers.append(layer)
        nindex.append({"l": "", "nr": "", "ni": ""})
    else:
        layers[id] = layer
    return
    

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