#%%
import pyspacemouse
import time
import multiprocessing

class SpacemouseInput:
    def __init__(
            self,
    ):
        success = pyspacemouse.open()
        if not success:
            print("No SpaceMouse Found")
            exit()

        self.manager = multiprocessing.Manager()
        self.current_spacemouse_state = self.manager.dict()

        self.read_spacemouse_process = multiprocessing.Process(target=self.read_spacemouse)
        self.read_spacemouse_process.start()
        print("Spacemouse Input Ready")
        
    def read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            self.current_spacemouse_state.update({
                'x': state.x,
                'y': state.y,
                'z': state.z,
                'roll': state.roll,
                'pitch': state.pitch,
                'yaw': state.yaw,
                'buttons': state.buttons,
            })
            # print(self.current_spacemouse_state)
            time.sleep(.001)

    @property
    def get_action(self):
        return self.current_spacemouse_state
    
    def close(self):
        pyspacemouse.close()
        self.read_spacemouse_process.terminate()
        self.read_spacemouse_process.join()
    
spacemouse_input = SpacemouseInput()
#%%
while True:
    print(spacemouse_input.get_action)
    time.sleep(1.0)