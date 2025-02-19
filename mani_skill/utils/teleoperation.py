import pyspacemouse
import multiprocessing
import time
import numpy as np

class SpacemouseInput:
    def __init__(
            self,
            mode:str='delta_pose',
            start_gripper_closed=True,
    ):
        success = pyspacemouse.open()
        if not success:
            print("No SpaceMouse Found")
            exit()

        supported_modes = ['delta_pose', 'target_pose']
        assert mode in ['delta_pose', 'target_pose'], f"mode should be in {supported_modes}, got {mode}"
        self.spacemouse_input_function = self.spacemouse_input_to_delta_pose if mode == 'delta_pose' else self.apply_spacemouse_input_to_target_pose

        self.mode = mode
        self.start_gripper_closed = start_gripper_closed
        self.gripper_action = -1 if start_gripper_closed else 1
        # self.translation_factor = 0.02
        # self.rotation_factor = 0.02

        self.translation_factor = 0.1
        self.rotation_factor = 0.15

        self.button_timeout = 0.4
        self.last_button_press_time = time.perf_counter()
        
        self.manager = multiprocessing.Manager()
        self.current_spacemouse_state = self.manager.dict()

        self.read_spacemouse_process = multiprocessing.Process(target=self.read_spacemouse)
        self.read_spacemouse_process.start()
    
    def reset(self):
        self.gripper_action = -1 if self.start_gripper_closed else 1

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
                
    def get_action(self):
        spacemouse_input = self.current_spacemouse_state
        return self.spacemouse_input_function(spacemouse_input)

    def spacemouse_input_to_delta_pose(self, spacemouse_event):
        delta_pose = np.zeros(7)
        # delta_pose[0] = -spacemouse_event.y * self.translation_factor
        delta_pose[1] = spacemouse_event['x'] * self.translation_factor
        delta_pose[2] = spacemouse_event['z'] * self.translation_factor
        delta_pose[3] = spacemouse_event['roll'] * self.rotation_factor
        # delta_pose[4] = spacemouse_event.pitch * self.rotation_factor
        # delta_pose[5] = spacemouse_event.yaw * self.rotation_factor
        if spacemouse_event['buttons'][1] and time.perf_counter() - self.last_button_press_time > self.button_timeout:
            self.gripper_action = -self.gripper_action
            self.last_button_press_time = time.perf_counter()
        delta_pose[6] = self.gripper_action

        return delta_pose

    def apply_spacemouse_input_to_target_pose(target_pose, spacemouse_event):
        target_pose[:3] += spacemouse_event[:3]*0.01
        target_pose[3:] += spacemouse_event[3:]*0.01
        return target_pose
    
    def close(self):
        pyspacemouse.close()
        self.read_spacemouse_process.terminate()
        self.read_spacemouse_process.join()