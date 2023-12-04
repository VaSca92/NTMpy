import eel
# Set web files folder and optionally specify which file types to check for eel.expose()

layers = []

@eel.expose
def setLayer(layer):
    layers.append(layer)


@eel.expose
def getLayer():
    return layers


eel.init('web', allowed_extensions=['.js', '.html', '.css'])
eel.start('html/main.html', size=(1080, 640), jinja_templates='html')    # Start
