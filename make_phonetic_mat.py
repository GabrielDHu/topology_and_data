import matplotlib.pyplot as plt
import numpy as np


xs = []
ys = []
for i in range(50):
    theta = np.random.uniform(0,1)
    epsilon_x = np.random.normal(0,0.06)
    epsilon_y = np.random.normal(0,0.06)

    x = np.cos(2*np.pi*theta) + epsilon_x
    y = np.sin(2*np.pi*theta) + epsilon_y

    xs.append(x)
    ys.append(y)

for j in range(25):
    theta = np.random.uniform(0,1)
    epsilon_x = np.random.normal(0,0.06)
    epsilon_y = np.random.normal(0,0.06)

    x = 1.5 + 0.5*np.cos(2*np.pi*theta) + epsilon_x
    y = 0.5*np.sin(2*np.pi*theta) + epsilon_y

    xs.append(x)
    ys.append(y)

for k in range(20):
    theta = np.random.uniform(0,1)
    epsilon_x = np.random.normal(0,0.03)
    epsilon_y = np.random.normal(0,0.03)

    x = 0.2*np.cos(2*np.pi*theta) + epsilon_x
    y = 1.2 + 0.2*np.sin(2*np.pi*theta) + epsilon_y

    xs.append(x)
    ys.append(y)


plt.scatter(xs,ys)
plt.show()


points = np.column_stack((xs, ys))
n = len(points)

from scipy.spatial.distance import cdist

dist_mat = cdist(points, points)


def plot_edges(ax, points, dist_mat, eps, color="black", alpha=0.3):
    n = len(points)
    for i in range(n):
        for j in range(i):
            if dist_mat[i, j] <= eps:
                ax.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    color="red",
                    linewidth=2,
                )

epsilons = [0.25, 0.4, 1.0]

for eps in epsilons:
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot points first
    ax.scatter(points[:, 0], points[:, 1], s=20, zorder=2)

    # then edges
    plot_edges(ax, points, dist_mat, eps)

    ax.set_title(f"Edges for distance ≤ {eps}")
    ax.set_aspect("equal")
    plt.show()


from ripser import ripser 
from persim import plot_diagrams
result = ripser(dist_mat, maxdim = 2, distance_matrix=True)

diagrams = result['dgms']


plot_diagrams(diagrams, show=True)

import gruut

text = "Isabella Isabelle"

for sentence in gruut.sentences(text, lang="en"):
            for word in sentence:
                phon = "".join(word.phonemes)
                print(phon)