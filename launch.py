import sys

# Check for required packages
REQUIRED_PACKAGES = ['numpy', 'eel', 'jinja2', 'tqdm', 'matplotlib']
MISSING_PACKAGES = []

print("="*110 + "\n")
print("Welcome to NTMpy GUI")
print("="*110 + "\n")
print("Checking required packages...\n")

for package in REQUIRED_PACKAGES:
    try:
        __import__(package)
        print(f"✓ {package} is installed")
    except ImportError:
        print(f"✗ {package} is NOT installed")
        MISSING_PACKAGES.append(package)

if MISSING_PACKAGES:
    print("\n" + "="*110)
    print("WARNING: The following required packages are missing:")
    for pkg in MISSING_PACKAGES: print(f"  - {pkg}")
    print("\nPlease install them using:\n")
    for package in MISSING_PACKAGES: print(f"  pip install {package}")
    print("="*110 + "\n")
    wait = input("Press Enter to continue.")
    sys.exit(1)
else:
    print("\n✓ All required packages are installed!\n")
    print("="*110 + "\n")

print("This software was developed by Valentino Scalera and Lukas Alber with the collaboration of the SU-UDCM Group")
print("Please, report any bug or feature request to valentino.scalera@uniparthenope.it")
print("The graphical user interface is starting, please wait\n")

import eel

import gui.py.variables # type: ignore
import gui.py.fun_material # type: ignore
import gui.py.fun_source # type: ignore
import gui.py.fun_files # type: ignore
import gui.py.fun_result # type: ignore
import gui.py.fun_fit # type: ignore
import gui.py.main # type: ignore


eel.init('.', allowed_extensions=['.js', '.html', '.css'])
eel.start('gui/html/page_main.html', size=(1000, 800), jinja_templates='gui/html', host='localhost', port=8000)    # Start

