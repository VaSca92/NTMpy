import eel
from init_variables import flags, laser, layers, nindex, sim, src


# Run Simulation #################################
@eel.expose
def run():
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