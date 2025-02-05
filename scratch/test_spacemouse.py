import pyspacemouse
import time
success = pyspacemouse.open()
if not success:
    print("No SpaceMouse Found")
    exit()
else:
    while True:
        event = pyspacemouse.read()
        print(event)
        time.sleep(0.1)