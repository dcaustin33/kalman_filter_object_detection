import matplotlib.pyplot as plt
import numpy as np

import object_tracking.kalman_filter.fake_data as fake_data
from object_tracking.kalman_filter.schema_2d import Kalman2DTracker, KalmanStateVector2D


def plot_trajectory(frame: np.ndarray, boxes1: list, boxes2: list = None):
    """Plots the trajectories with different colors
    
    Args:
        frame: The frame to plot on
        boxes1: First trajectory (original measurements) 
        boxes2: Second trajectory (Kalman predictions) if provided
    """
    # Plot original trajectory in red
    for box in boxes1:
        frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1
        
    # Plot Kalman predictions in blue if provided
    if boxes2 is not None:
        for box in boxes2:
            frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 2
            
    plt.imshow(frame, cmap='Set1')  # Using Set1 colormap for distinct colors
    plt.legend(['Original', 'Kalman Filter'])
    plt.show()
    return frame


if __name__ == "__main__":
    # Generate test data
    all_boxes = fake_data.generate_fake_data()
    frame = np.zeros((1000, 1000))

    # Initialize Kalman filter
    state = KalmanStateVector2D(x=all_boxes[0][0], y=all_boxes[0][1], vx=0, vy=0)
    tracker = Kalman2DTracker(state, state_noise_std=.01, measurement_noise_std=.01)

    # Track and store Kalman predictions
    kalman_states = []
    for box in all_boxes:
        tracker.update(np.array([box[0], box[1]]), dt=1)
        kalman_states.append([int(tracker.state.x), int(tracker.state.y), 1, 1])

    # Plot both trajectories
    plot_trajectory(frame, all_boxes, kalman_states)