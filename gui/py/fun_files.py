import eel
import json, os
from numpy import savez, load
from gui.py.variables import flags, laser, layers, nindex, current_file, current_data, current_path, time, out, layer_state, outdir, fit, store


# Save File #####################################
@eel.expose
def saveFile(filename="ntmpy_save", path="./data/models/"):
    data_to_save = {
        "flags"  :  flags,
        "laser"  :  laser,
        "layers" : layers,
        "nindex" : nindex,
        "time"   :   time,
        "expdata": current_data[0]
    }
    current_file[0] = filename
    try:
        json.dump(data_to_save, open(path + filename + ".json", 'w'), indent=4)
        if flags["result_set"] and not flags["spin_temp"]:
            outdir[0] = "../output/" if path == "./data/models/" else ""
            savez(path + outdir[0] + filename, x = out["x"], t=out["t"], Te=out["T"][0], Tl=out["T"][1])
            return("Successfully saved to " + filename + ".json and " + filename + ".npz")
        elif flags["result_set"] and flags["spin_temp"]:
            outdir[0] = "../output/" if path == "./data/models/" else ""
            savez(path + outdir[0] + filename, x = out["x"], t=out["t"], Te=out["T"][0], Tl=out["T"][1], Ts=out["T"][2])
            return("Successfully saved to " + filename + ".json and " + filename + ".npz")
        return("Successfully saved to " + path + filename + ".json")
    except Exception as e:
        return("Error saving file: "+ str(e))


# Load File #####################################
@eel.expose
def loadFile(filename="ntmpy_save"):

    path = current_path[0]
    print("loading file: ", path + filename)

    try:
        with open(path + filename, 'r') as f:
            data_loaded = json.load(f)

        flags.clear()
        flags.update(data_loaded.get("flags", {}))

        time.clear()
        time.update( data_loaded.get("time", {}))

        laser.clear()
        laser.update(data_loaded.get("laser", {}))

        layers.clear()
        layers.extend(data_loaded.get("layers", []))

        nindex.clear()
        nindex.extend(data_loaded.get("nindex", []))

        current_file[0] = filename.split(".")[0]
        current_data[0] = data_loaded.get("expdata", "")

        store.clear()
        store.update({  "time_sim" : None,
                        "temp_sim" : None,
                        "time_exp" : None,
                        "temp_exp" : None,
                        "residual" : []})
    
        fit.clear()
        fit.update({    "point"   : [],
                        "value"   : [],
                        "coeff"   : [],
                        "target"  : [],
                        "function": None,
                        "weight"  : None,
                        "data"    : None,
                        "init"    : False})


        if flags["result_set"] and not flags["spin_temp"]:
            outdir[0] = "../output/" if path == "./data/models/" else ""
            try: 
                with load(path + outdir[0] + current_file[0] + ".npz") as data:
                    out["t"] = data["t"]
                    out["x"] = data["x"]
                    out["T"] = [data["Te"], data["Tl"]]
                    return("Successfully loaded from " + current_file[0] + ".json and " + current_file[0] + ".npz")
            except FileNotFoundError:
                return("Successfully loaded from " + current_file[0] + ".json")
        elif flags["result_set"] and flags["spin_temp"]:
            outdir[0] = "../output/" if path == "./data/models/" else ""
            try:
                with load(path + outdir[0] + current_file[0] + ".npz") as data:
                    out["t"] = data["t"]
                    out["x"] = data["x"]
                    out["T"] = [data["Te"], data["Tl"], data["Ts"]]
                    return("Successfully loaded from " + current_file[0] + ".json and " + current_file[0] + ".npz")
            except FileNotFoundError:
                return("Successfully loaded from " + current_file[0] + ".json")
        else: 
            return("Successfully loaded from " + current_file[0] + ".json")
    except FileNotFoundError:
        return("Error: File " + filename + " not found.")
    except Exception as e:
        return("Error loading file: " + str(e))

# Load File #####################################
@eel.expose
def newFile():

    flags.clear()
    laser.clear()
    layers.clear()
    nindex.clear()
    layer_state.clear()
    time.clear()
    fit.clear()
    store.clear()
  
    laser.update({  "energy":       None,
                    "fwhm":         None,
                    "delay":        None,
                    "wavelength":   None,
                    "angle":        None,
                    "polarization": None})
                    
    flags.update({  "spin_temp" : False,
                    "reflection":  True,
                    "substrate" : False,
                    "source_set": False,
                    "almost_set": False,
                    "layers_set": False,
                    "result_set": False})

    time.update({   "simulation" : 0,
                    "computation": 0})

    store.update({  "time_sim" : None,
                    "temp_sim" : None,
                    "time_exp" : None,
                    "temp_exp" : None,
                    "residual" : []})

    fit.update({    "point"   : [],
                    "value"   : [],
                    "coeff"   : [],
                    "target"  : [],
                    "function": None,
                    "weight"  : None,
                    "data"    : None,
                    "init"    : False})

    out.clear()


    current_file[0] = "untitled"
    current_data[0] = ""


# Explore Files #################################
@eel.expose
def exploreFiles(path):
    if not path:
        path = loadPath()
    if path == "./data/models/":
        outdir[0] = "../output/"
    else: 
        outdir[0] = ""
    try:
        files = [f for f in os.listdir(path) if f.endswith('.json')]
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        current_path[0] = path
        return folders + files
    except Exception as e:
        print("Error exploring files: " + str(e))

# Delete File ###################################
@eel.expose
def deleteFile(filename, path = "./data/models/"):
    try:
        os.remove(path + filename)
        if flags["result_set"]:
            os.remove(path + outdir[0] + filename.split(".")[0] + ".npz")
        return("Successfully deleted " + filename + ".json and " + filename + ".npz")
    except Exception as e:
        return("Error deleting file: " + str(e))

# Save and Load Preferences #####################
@eel.expose
def savePath(path):
    with open("./gui/pref/last_path.txt", "w") as f:
        f.write(path)

@eel.expose
def loadPath():
    with open("./gui/pref/last_path.txt", "r") as f:
        try:
            return f.read()
        except FileNotFoundError:
            return "./data/models/"

@eel.expose
def getFilename():
    return(current_file)


