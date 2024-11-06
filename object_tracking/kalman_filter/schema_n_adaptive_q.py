"""
Implementation of a ND Kalman Filter for tracking objects

Assumptions:
    - We transition one step at a time otherwise f needs to be updated, therefore every step needs a prediction
    
"""

"""
Adaptive Q addition is to add an update q func that takes into account the innovation
"""


from dataclasses import dataclass

import numpy as np
from scipy import stats


class KalmanStateVectorNDAdaptiveQ:
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
        self.f[: states.shape[0], states.shape[0] :] = np.eye(states.shape[0])

    def initialize_covariance(self, noise_std: float) -> None:
        self.cov = np.eye(self.state_matrix.shape[0]) * noise_std**2

    def predict_next_state(self, dt: float) -> None:
        self.state_matrix = self.f @ self.state_matrix
        self.predict_next_covariance(dt)

    def predict_next_covariance(self, dt: float) -> None:
        self.cov = self.f @ self.cov @ self.f.T + self.q

    def __add__(self, other: np.ndarray) -> np.ndarray:
        return self.state_matrix + other

    def update_q(
        self, innovation: np.ndarray, kalman_gain: np.ndarray, alpha: float = 0.98
    ) -> None:
        innovation = innovation.reshape(-1, 1)
        self.q = (
            alpha * self.q
            + (1 - alpha) * kalman_gain @ innovation @ innovation.T @ kalman_gain.T
        )


class KalmanNDTrackerAdaptiveQ:

    def __init__(
        self,
        state: KalmanStateVectorNDAdaptiveQ,
        R: float,  # R
        Q: float,  # Q
        h: np.ndarray = None,
    ) -> None:
        self.state = state
        self.state.initialize_covariance(Q)
        self.predicted_state = None
        self.previous_states = []
        self.previous_measurements = []
        self.h = np.eye(self.state.state_matrix.shape[0]) if h is None else h
        self.R = np.eye(self.h.shape[0]) * R**2

    def predict(self, dt: float) -> None:
        self.previous_states.append(self.state)
        self.state.predict_next_state(dt)

    def update_covariance(self, gain: np.ndarray) -> None:
        self.state.cov -= gain @ self.h @ self.state.cov

    def update(
        self, measurement: np.ndarray, dt: float = 1, predict: bool = True
    ) -> None:
        """Measurement will be a x, y position"""
        self.previous_measurements.append(measurement)
        assert dt == 1, "Only single step transitions are supported due to F matrix"
        if predict:
            self.predict(dt=dt)
        innovation = measurement - self.h @ self.state.state_matrix
        gain_invertible = self.h @ self.state.cov @ self.h.T + self.R
        gain_inverse = np.linalg.inv(gain_invertible)
        gain = self.state.cov @ self.h.T @ gain_inverse

        new_state = self.state.state_matrix + gain @ innovation

        self.update_covariance(gain)
        self.state.update_q(innovation, gain)
        self.state.state_matrix = new_state

    def compute_mahalanobis_distance(self, measurement: np.ndarray) -> float:
        innovation = measurement - self.h @ self.state.state_matrix
        return np.sqrt(
            innovation.T
            @ np.linalg.inv(
                self.h @ self.state.cov @ self.h.T + self.measurement_noise_std
            )
            @ innovation
        )

    def compute_p_value(self, distance: float) -> float:
        return 1 - stats.chi2.cdf(distance, df=self.h.shape[0])

    def compute_p_value_from_measurement(self, measurement: np.ndarray) -> float:
        """Returns the probability that the measurement is consistent with the predicted state"""
        distance = self.compute_mahalanobis_distance(measurement)
        return self.compute_p_value(distance)
