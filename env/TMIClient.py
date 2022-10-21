from tminterface.client import Client
from tminterface.interface import TMInterface
from threading import Thread, Lock
import time


class SimStateClient(Client):
    def __init__(self):
        self.sim_state = None

    def on_run_step(self, iface, _time: int):
        self.sim_state = iface.get_simulation_state()


class ThreadedClient:
    def __init__(self) -> None:
        self.TMIClient = SimStateClient()
        self._client_thread = Thread(target=self.client_thread, daemon=True)
        self._lock = Lock()
        self.data = None
        self._client_thread.start()
        self.iface = TMInterface()

    def client_thread(self):
        client = SimStateClient()
        print("ok")

        self.iface.register(client)
        while self.iface.running:
            time.sleep(0)
            self._lock.acquire()
            self.data = client.sim_state
            self._lock.release()


if __name__ == "__main__":
    simthread = ThreadedClient()
    while True:
        print(simthread.data)
