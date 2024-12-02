Kalman filter implementation specifically built for tracking object detection boxes.


# Adaptive Q
The most commonly used kalman filter. Q is adaptive and will learn during tracking.

Usage for 2d center position tracking
```
state = KalmanStateVectorNDAdaptiveQ(
    states=np.array([all_boxes[0][0], all_boxes[0][1]]), velocities=np.array([0, 0])
)
# H matrix only measure position
h_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
tracker = KalmanNDTrackerAdaptiveQ(state, Q=1, R=2.5, h=h_matrix)
```

Usage for object detection box tracking
```
state = KalmanStateVectorNDAdaptiveQ(
    states=np.array([all_boxes[0][0], all_boxes[0][1], all_boxes[0][2], all_boxes[0][3]]), velocities=np.array([0, 0, 0, 0])
)
# H matrix. Measurement is just box dimensions
h_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]])
# set Q and R as needed based on noise estimates. Q is adaptive and will learn during tracking.
tracker = KalmanNDTrackerAdaptiveQ(state, Q=1, R=2.5, h=h_matrix)
```