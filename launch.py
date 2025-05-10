import sys
sys.path.insert(0, './gui/python')

import eel


import init_variables # type: ignore
import fun_material # type: ignore
import fun_source # type: ignore

# Save all #######################################
@eel.expose
def save():
     pass


eel.init('.', allowed_extensions=['.js', '.html', '.css'])
eel.start('gui/html/main.html', size=(1120, 720), jinja_templates='gui/html')    # Start

