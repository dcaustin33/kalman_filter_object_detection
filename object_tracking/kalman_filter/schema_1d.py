"""
Implementation of a 1D Kalman Filter for tracking a single object

Assumptions:
    - There is no Q noise
    - We can independently track x, y so we only need to implement a 1D version
"""

from dataclasses import dataclass

import numpy as np


class KalmanStateVector1D:
    x: float
    vx: float
    cov: np.ndarray

    def __init__(self, x: float, vx: float) -> None:
        self.x = x
        self.vx = vx
        self.cov = None

    def initialize_covariance(self, noise_std: float) -> None:
        self.cov = np.array([[noise_std**2, 0], [0, noise_std**2]])

    def predict_next_state(self, dt: float) -> None:
        self.x = self.x + self.vx * dt
        self.predict_next_covariance(dt)

    def predict_next_covariance(self, dt: float) -> None:
        # for right now we are assuming no Q noise
        self.cov[0][0] = self.cov[0][0] + self.cov[1][1] * dt**2
        self.cov[0][1] = self.cov[0][1] + self.cov[1][1] * dt
        self.cov[1][0] = self.cov[1][0] + self.cov[1][1] * dt
        self.cov[1][1] = self.cov[1][1]
        
    def __add__(self, other: np.ndarray) -> np.ndarray:
        return np.array([self.x + other[0][0], self.vx + other[1][0]])


class Kalman1DTracker:

    def __init__(
        self,
        state: KalmanStateVector1D,
        state_noise_std: float,
        measurement_noise_std: float,
    ) -> None:
        self.state = state
        self.state.initialize_covariance(state_noise_std)
        self.predicted_state = None
        self.measured_state = None
        self.measurement_noise_std = measurement_noise_std
        self.previous_states = []

    def predict(self, dt: float) -> None:
        # may need to make sure this is copied and not just a reference
        self.previous_states.append(self.state)
        self.state.predict_next_state(dt)

    def update(self, measurement: float, dt: float = 1) -> None:
        self.predict(dt=1)
        self.measured_state = measurement

        innovation = self.measured_state - self.state.x
        gain_first_term = 1 / (
            self.state.cov[0][0] + self.measurement_noise_std**2
        )
        gain_second_term = np.array(
            [self.state.cov[0][0], self.state.cov[0][1]]
        ).reshape(-1, 1)
        gain = gain_first_term * gain_second_term

        new_state = self.state + gain * innovation
        self.state.x = new_state[0]
        self.state.vx = new_state[1]
