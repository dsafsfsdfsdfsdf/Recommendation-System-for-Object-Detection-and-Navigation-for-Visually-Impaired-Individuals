import numpy as np
import cv2
import heapq
import pyttsx3  # Offline TTS library


class Node:
    """
    A* pathfinding node.
    Stores position, path cost so far, and priority for the queue.
    """
    def __init__(self, x, y, cost, priority):
        self.x = x
        self.y = y
        self.cost = cost
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority


def astar_path_planning(map_binary, start, goal):
    """
    Perform A* pathfinding on a binary map.

    Args:
        map_binary (np.ndarray): 1 = free, 0 = obstacle.
        start (tuple): (row, col)
        goal (tuple): (row, col)

    Returns:
        list: Ordered list of (row, col) path points
    """
    h, w = map_binary.shape
    came_from = {}
    cost_so_far = {}
    queue = []

    heapq.heappush(queue, Node(start[0], start[1], 0, 0))
    came_from[start] = None
    cost_so_far[start] = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current = heapq.heappop(queue)
        if (current.x, current.y) == goal:
            break

        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy
            if 0 <= nx < h and 0 <= ny < w and map_binary[nx, ny] == 1:
                new_cost = cost_so_far[(current.x, current.y)] + 1
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + abs(goal[0] - nx) + abs(goal[1] - ny)
                    heapq.heappush(queue, Node(nx, ny, new_cost, priority))
                    came_from[(nx, ny)] = (current.x, current.y)

    # Reconstruct the shortest path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return []
    path.append(start)
    path.reverse()
    return path


def draw_path_on_bev(map_binary, path):
    """
    Overlay a path on a BEV (binary) map.

    Args:
        map_binary (np.ndarray): 1 = free, 0 = obstacle
        path (list): list of (row, col) positions

    Returns:
        np.ndarray: Color BGR image with green path overlay
    """
    bev_color = cv2.cvtColor((map_binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for y, x in path:
        cv2.circle(bev_color, (x, y), 1, (0, 255, 0), -1)
    return bev_color


def simulate_and_speak_alerts(bev_map, path, scan_data, fov_deg=60, max_range=2.0):
    """
    Simulate following the path and announce nearby obstacles.

    Args:
        bev_map (np.ndarray): 1 = free, 0 = obstacle
        path (list): Path points
        scan_data (list of (angle, norm_distance)): Pseudo LiDAR scan
        fov_deg (float): Field of view of scan
        max_range (float): Maximum scan distance in meters

    Returns:
        list of spoken alert strings
    """
    scan_data = list(scan_data)
    h, w = bev_map.shape
    alerts = []
    angle_list = np.linspace(-fov_deg / 2, fov_deg / 2, w)

    for i, (y, x) in enumerate(path):
        if i % 5 != 0 or x >= len(scan_data):
            continue

        submap = bev_map[max(0, y - 2):y + 2, max(0, x - 2):x + 2]
        if np.any(submap == 0):
            angle = angle_list[x]
            dist_norm = float(scan_data[x][1])
            dist_norm = np.clip(dist_norm, 0.01, 1.0)

            # Nonlinear mapping for better near-distance precision
            dist_meter = (dist_norm ** 1.8) * max_range

            direction = "ahead"
            if angle < -10:
                direction = "to your left"
            elif angle > 10:
                direction = "to your right"

            alert = f"Obstacle {dist_meter:.1f} meters {direction}."
            speak(alert)
            alerts.append(alert)

    return alerts


def speak(text):
    """
    Speak out the given text using offline English TTS.

    Auto-selects a likely English voice from system.
    """
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    found = False
    for v in voices:
        if "David" in v.id or "Zira" in v.id or "en_US" in v.id or "english" in v.id.lower():
            engine.setProperty('voice', v.id)
            found = True
            break
    if not found:
        print("[WARN] English voice not found, using system default.")

    engine.say(text)
    engine.runAndWait()
