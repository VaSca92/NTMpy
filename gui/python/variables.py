import sys
sys.path.insert(0, './code')
from Sim2T import Sim2T # type: ignore
from Source import source # type: ignore

global src; src = source()
global data; data = {"src": None}
global sim; sim = Sim2T()


global layers; layers = []
global nindex; nindex = []

global flags; flags = {"reflection": False, "source_set": False}
global laser; laser = {"energy": 0, "fwhm": 0, "delay": 0}


