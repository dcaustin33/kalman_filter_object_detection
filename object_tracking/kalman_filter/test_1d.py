import matplotlib.pyplot as plt
import numpy as np

import object_tracking.kalman_filter.fake_data as fake_data
from object_tracking.kalman_filter.schema_1d import KalmanStateVector1D, Kalman1DTracker


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
            
    frame = frame[:50, :50]
    plt.imshow(frame, cmap='Set1')  # Using Set1 colormap for distinct colors
    plt.legend(['Original', 'Kalman Filter'])
    plt.show()
    return frame


if __name__ == "__main__":
    # Generate test data
    all_boxes = fake_data.generate_fake_data()
    frame = np.zeros((400, 400))

    # Initialize Kalman filter
    state_x = KalmanStateVector1D(x=all_boxes[0][0], vx=0)
    state_y = KalmanStateVector1D(x=all_boxes[0][1], vx=0)
    tracker_x = Kalman1DTracker(state_x, state_noise_std=.01, measurement_noise_std=.01)
    tracker_y = Kalman1DTracker(state_y, state_noise_std=.01, measurement_noise_std=.01)

    # Track and store Kalman predictions
    kalman_states = []
    for box in all_boxes:
        tracker_x.update(box[0], dt=1)
        tracker_y.update(box[1], dt=1)
        kalman_states.append([int(tracker_x.state.x), int(tracker_y.state.x), 1, 1])

    # Plot both trajectories
    plot_trajectory(frame, all_boxes, kalman_states)