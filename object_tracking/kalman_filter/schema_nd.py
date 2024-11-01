"""
Implementation of a ND Kalman Filter for tracking objects

Assumptions:
    - We transition one step at a time otherwise f needs to be updated, therefore every step needs a prediction
    
"""

from dataclasses import dataclass

import numpy as np


class KalmanStateVectorND:
    states: np.ndarray
    velocities: np.ndarray
    cov: np.ndarray

    def __init__(self, states: np.ndarray, velocities: np.ndarray) -> None:
        self.states = states
        self.velocities = velocities
        self.state_matrix = np.concatenate([self.states, self.velocities])
        self.q = np.eye(self.state_matrix.shape[0]) * 0.01
        self.cov = None
        # assumes a single step transition
        self.f = np.eye(self.state_matrix.shape[0])
        self.f[:states.shape[0], states.shape[0]:] = np.eye(states.shape[0])

    def initialize_covariance(self, noise_std: float) -> None:
        self.cov = np.eye(self.state_matrix.shape[0]) * noise_std**2

    def predict_next_state(self, dt: float) -> None:
        self.state_matrix = self.f @ self.state_matrix
        self.predict_next_covariance(dt)

    def predict_next_covariance(self, dt: float) -> None:
        self.cov = self.f @ self.cov @ self.f.T + self.q

    def __add__(self, other: np.ndarray) -> np.ndarray:
        return self.state_matrix + other


class KalmanNDTracker:

    def __init__(
        self,
        state: KalmanStateVectorND,
        state_noise_std: float, # R
        measurement_noise_std: float, # Q
        h: np.ndarray = None,
    ) -> None:
        self.state = state
        self.state.initialize_covariance(state_noise_std)
        self.predicted_state = None
        self.previous_states = []
        self.h = np.eye(self.state.state_matrix.shape[0]) if h is None else h
        self.measurement_noise_std = np.eye(self.h.shape[0]) * measurement_noise_std**2

    def predict(self, dt: float) -> None:
        self.previous_states.append(self.state)
        self.state.predict_next_state(dt)

    def update_covariance(self, gain: np.ndarray) -> None:
        self.state.cov -= gain @ self.h @ self.state.cov

    def update(self, measurement: np.ndarray, dt: float = 1, predict: bool = True) -> None:
        """Measurement will be a x, y position"""
        assert dt == 1, "Only single step transitions are supported due to F matrix"
        if predict:
            self.predict(dt=dt)
        innovation = measurement - self.h @ self.state.state_matrix
        gain_invertible = (
            self.h @ self.state.cov @ self.h.T + self.measurement_noise_std
        )
        gain_inverse = np.linalg.inv(gain_invertible)
        gain = self.state.cov @ self.h.T @ gain_inverse

        new_state = self.state.state_matrix + gain @ innovation
        self.update_covariance(gain)
        self.state.state_matrix = new_state
        
    def compute_mahalanobis_distance(self, measurement: np.ndarray) -> float:
        innovation = measurement - self.h @ self.state.state_matrix
        return np.sqrt(innovation.T @ np.linalg.inv(self.h @ self.state.cov @ self.h.T + self.measurement_noise_std) @ innovation)
