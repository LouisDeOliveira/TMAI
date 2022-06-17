from tminterface.interface import Client
from tminterface.client import run_client

class Myclient(Client):
    def __init__(self,):
        super().__init__()
        self.total_reward = 0

    def on_registered(self, iface):
        print("Registered to TMInterface")

    def on_simulation_step(self, iface, _time: int):
        print(f"Simulation step, time = {_time}")

    def on_run_step(self, iface, _time: int):
        if _time%50 == 0:

            iface.execute_command(f"{_time}-{_time+50} gas -30000")
        if _time%500 == 0:
            print(f"time : {_time}, speed : {iface.get_simulation_state().display_speed}")
            YPRinfo = iface.get_simulation_state().yaw_pitch_roll
            print(f" yaw: {YPRinfo[0]}, pitch: {YPRinfo[1]}, roll: {YPRinfo[2]}")
            self.total_reward += self.reward(iface)
            print(f"reward: {self.reward(iface)}")
            print(f"total reward: {self.total_reward}")

    def reward(self, iface):
        speed_reward = iface.get_simulation_state().display_speed/500
        roll_reward = - abs(iface.get_simulation_state().yaw_pitch_roll[2])/3.15
        constant_reward = -0.3

        return speed_reward + roll_reward + constant_reward


    
client = Myclient()

run_client(client)

