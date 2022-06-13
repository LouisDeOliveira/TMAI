from errno import EMSGSIZE
from http import client
from tminterface import constants
from tminterface.interface import TMInterface, Client
from tminterface.client import run_client

class Myclient(Client):
    def on_registered(self, iface):
        print("Registered to TMInterface")
    def on_simulation_step(self, iface, _time: int):
        print(f"Simulation step, time = {_time}")
    def on_run_step(self, iface, _time: int):
        if _time%500 == 0:
            print(f"time : {_time}, speed : {iface.get_simulation_state().display_speed}")
            YPRinfo = iface.get_simulation_state().yaw_pitch_roll
            print(f" yaw: {YPRinfo[0]}, pitch: {YPRinfo[1]}, roll: {YPRinfo[2]}")
            print(f"reward: {self.reward(iface)}")
    def reward(self, iface):
        speed_reward = iface.get_simulation_state().display_speed/500
        roll_reward = - abs(iface.get_simulation_state().yaw_pitch_roll[2])/3.15
        constant_reward = -0.3

        return speed_reward + roll_reward + constant_reward


            
        
    @staticmethod
    def __get_int(buffer: bytearray, offset: int) -> int:
        return int.from_bytes(buffer[offset:offset+4], byteorder='little')
    
client = Myclient()

run_client(client)

