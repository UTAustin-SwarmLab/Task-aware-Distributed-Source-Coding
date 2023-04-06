import numpy as np
import matplotlib.pyplot as plt
from typing import List

def delta_cover(radius: float, delta: float) -> List[np.ndarray]:
    """
    Finds an optimal delta-cover for a 2-D l2-ball of radius r.
    Returns a list of centers of delta-balls.
    """
    centers = []
    x = -radius
    while x <= radius:
        y = -radius
        while y <= radius:
            if np.linalg.norm([x,y]) <= radius:
                centers.append(np.array([x,y]))
            y += delta
        x += delta

    theta = np.arcsin(delta / radius / 2) * 4
    theta = theta / np.pi * 180
    for i in np.arange(0, 360, theta):
        centers.append(np.array([np.cos(i/180*np.pi), np.sin(i/180*np.pi)]))

    return centers

r = 1
delta_values = [0.1, 0.01, 0.001, 0.0001]
# delta_values.reverse()

for delta in delta_values:
    centers = delta_cover(r, delta)
    print(f"Delta: {delta}, number of centers: {len(centers)}")
    
    # Plot the delta-cover
    # fig, ax = plt.subplots(figsize=(9, 9))
    # ax.set_aspect('equal')
    # circle = plt.Circle((0,0), r, fill=False, edgecolor='b')
    # ax.add_artist(circle)
    # for center in centers:
    #     delta_circle = plt.Circle(center, delta, fill=False, edgecolor='r')
    #     ax.add_artist(delta_circle)
    # plt.xlim(-1.15, 1.15)
    # plt.ylim(-1.15, 1.15)
    # plt.show()  