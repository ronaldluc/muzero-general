"""
"Drone" that has API compatible with our "Robot", except this one just flies wherever it's sent.
Kinda like a drone that hovers.
"""
from simulation.robot_dynamics import Robot
import numpy as np

class Drone(Robot):
    def step(self, dt):
        self.hull.position[0] += self.wheels[2].gas/2
        self.hull.position[1] += self.wheels[3].gas/2


class DroneNoInertion(Drone):
    def gas_wheel(self, gas, wheel_id):
        wheel = self.wheels[wheel_id]
        gas = np.clip(gas, -1, 1)
        wheel.gas += gas
    
    def brake(self, intensity):
        for w in self.wheels:
            w.gas = 0
