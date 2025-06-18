import numpy as np
import matplotlib.pyplot as plt    

def plot():
    data = np.load("action_angle/cartesian.npz")

    x = data['x']
    y = data['y']

    plt.scatter(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot()