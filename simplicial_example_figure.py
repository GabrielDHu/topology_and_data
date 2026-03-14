import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# We will first create a set of data points which are uniformly generated from a
# large and small circle, as well as some 2-dimensional normal noise 
rng = np.random.default_rng(42)


x = np.sin(0)
y = np.cos(0)
circ1 = np.array([x,y])


circ_1_data = rng.uniform(0,1,40)
mean = [0, 0]
cov = [[0.01, 0],
       [0, 0.01]]

circ_1_noise = rng.multivariate_normal(mean, cov, 40)


for i in range(40):
    x = circ_1_data[i]
    noise = circ_1_noise[i,:]

    point = [np.sin(2*np.pi*x),np.cos(2*np.pi*x)] + noise

    circ1 = np.vstack((circ1, point))



circ_2_data = rng.uniform(0,1,15)
mean = [0, 0]
cov = [[0.0002, 0],
       [0, 0.0002]]

circ_2_noise = rng.multivariate_normal(mean, cov, 15)


for i in range(15):
    x = circ_2_data[i]
    noise = circ_2_noise[i,:]

    point = np.array([0.3*np.sin(2*np.pi*x),0.3*np.cos(2*np.pi*x)]) + np.array([-0.2,-1]) + noise

    circ1 = np.vstack((circ1, point))


# Now, all of our data points are stored in 