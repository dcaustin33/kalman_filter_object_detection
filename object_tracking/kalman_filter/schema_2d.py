"""
Implementation of a 1D Kalman Filter for tracking a single object

Assumptions:
    - There is no Q noise
    - We transition one step at a time otherwise f needs to be updated
    
"""

from dataclasses import dataclass

import numpy as np


class KalmanStateVector2D:
    x: float
    y: float
    vx: float
    vy: float
    cov: np.ndarray

    def __init__(self, x: float, y: float, vx: float, vy: float) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.q = np.zeros((4, 4))
        self.cov = None
        # assumes a single step transition
        self.f = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])

    def initialize_covariance(self, noise_std: float) -> None:
        self.cov = np.array(
            [
                [noise_std**2, 0, 0, 0],
                [0, noise_std**2, 0, 0],
                [0, 0, noise_std**2, 0],
                [0, 0, 0, noise_std**2],
            ]
        )

    def predict_next_state(self, dt: float) -> None:
        self.x = self.x + self.vx * dt
        self.y = self.y + self.vy * dt
        self.predict_next_covariance(dt)

    def predict_next_covariance(self, dt: float) -> None:
        self.cov = self.f @ self.cov @ self.f.T + self.q

    def __add__(self, other: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self.x + other[0][0],
                self.y + other[1][0],
                self.vx + other[2][0],
                self.vy + other[3][0],
            ]
        )

    @property
    def state_matrix(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy])


class Kalman2DTracker:

    def __init__(
        self,
        state: KalmanStateVector2D,
        state_noise_std: float,
        measurement_noise_std: float,
    ) -> None:
        self.state = state
        self.state.initialize_covariance(state_noise_std)
        self.predicted_state = None
        self.measured_state = None
        self.measurement_noise_std = np.array(
            [[measurement_noise_std, 0], [0, measurement_noise_std]]
        )
        self.previous_states = []
        self.h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def predict(self, dt: float) -> None:
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
        self.state.y = new_state[1]
        self.state.vx = new_state[2]
        self.state.vy = new_state[3]
