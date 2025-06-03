import cv2
import numpy as np

def show_bev_map(bev_map):
    """
    Display the BEV map as a pseudo-colored image using OpenCV.

    Args:
        bev_map (np.ndarray): 2D BEV occupancy map
    """
    # Resize for better visibility
    vis = cv2.resize(bev_map, (512, 128), interpolation=cv2.INTER_NEAREST)

    # Normalize values to 0–255 in case it's not
    if vis.max() <= 1:
        vis = (vis * 255).astype(np.uint8)

    # Apply colormap to enhance visibility
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    # Show the BEV map
    cv2.imshow("BEV Map", vis_color)
    cv2.waitKey(1)
