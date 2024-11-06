import random

import numpy as np


def generate_fake_data(steps: int = 100) -> list:
    """Generates fake kalman data used for testing"""
    random.seed(42)

    position_0 = (10, 10)
    width = 2
    height = 2


    random.seed(42)
    
    # boxes will be in cx, cy, h, w format
    all_boxes = []
    current_position = position_0
    for i in range(steps):
        velocity_x = random.randint(1, 10)
        velocity_y = random.randint(1, 10)
        width_noise = random.randint(-2, 2)
        height_noise = random.randint(-2, 2)
        
        current_position = [current_position[0] + velocity_x, current_position[1] + velocity_y]
        current_width = width + width_noise
        current_height = height + height_noise

        all_boxes.append([current_position[0], current_position[1],  current_width, current_height])
        
    return all_boxes
