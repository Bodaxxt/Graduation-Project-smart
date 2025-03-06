import os
import subprocess
import eel
from engine.features import *
from engine.command import *

# Initialize Eel with the 'www' folder containing your web files
eel.init("www")

# Define a function to start the app
def start_app():
    # Start the Eel app, specifying Microsoft Edge as the browser
    eel.start('index.html')

# Call playAssistantSound after starting the app
@eel.expose
def init_app():
    playAssistantSound()

# Open Microsoft Edge in app mode with the Eel app
os.system(r'start msedge.exe --app="file:///E:\chatbooot\www\index.html"')

# Start the app
start_app()