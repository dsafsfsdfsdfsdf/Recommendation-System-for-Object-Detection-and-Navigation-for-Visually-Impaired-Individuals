# fusion/simple_fusion.py

import numpy as np

def scan_to_bev(scan_data, bev_width=256, bev_height=64, fov_deg=60):
    """
    Convert scan data (angle, normalized distance) to a 2D BEV occupancy grid.

    Args:
        scan_data (list): List of (angle_rad, distance_normalized)
        bev_width (int): Horizontal resolution
        bev_height (int): Vertical resolution
        fov_deg (float): Field of view of scan in degrees

    Returns:
        np.ndarray: 2D BEV map (bev_height x bev_width), values 0 or 255
    """
    bev = np.zeros((bev_height, bev_width), dtype=np.uint8)
    fov_rad = np.deg2rad(fov_deg)

    for angle, dist in scan_data:
        if not (0.01 < dist <= 1.0):
            continue  # skip invalid values

        # angle ∈ [-fov/2, +fov/2] → x ∈ [0, bev_width]
        x = int(((angle + fov_rad / 2) / fov_rad) * bev_width)
        y = int((1.0 - dist) * bev_height)

        if 0 <= x < bev_width and 0 <= y < bev_height:
            bev[y, x] = 255

    return bev
