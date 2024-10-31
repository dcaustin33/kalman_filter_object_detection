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
        self.q = np.zeros((2, 2))
        self.f = np.array([[1, 1], [0, 1]])

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
    
    @property
    def state_matrix(self) -> np.ndarray:
        return np.array([self.x, self.vx])


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
        self.measurement_noise_std = np.array([measurement_noise_std])
        self.previous_states = []
        self.h = np.array([[1, 0]])

    def predict(self, dt: float) -> None:
        # may need to make sure this is copied and not just a reference
        self.previous_states.append(self.state)
        self.state.predict_next_state(dt)
        
    def update_covariance(self, gain: np.ndarray) -> None:
        self.state.cov -= gain @ self.h @ self.state.cov

    def update(self, measurement: np.ndarray, dt: float = 1) -> None:
        """Measurement will be a x, y position"""
        assert dt == 1, "Only single step transitions are supported due to F matrix"
        self.predict(dt=dt)
        innovation = measurement - self.h @ self.state.state_matrix
        gain_invertible = (
            self.h @ self.state.cov @ self.h.T + self.measurement_noise_std**2
        )
        gain_inverse = np.linalg.inv(gain_invertible)
        gain = self.state.cov @ self.h.T @ gain_inverse

        new_state = self.state.state_matrix + gain @ innovation
        self.update_covariance(gain)
        self.state.x = new_state[0]
        self.state.vx = new_state[1]
        
