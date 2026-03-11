import numpy as np
import matplotlib.pyplot as plt

# example function
def f(x):
    return np.sin(x)

x = np.linspace(0, 10, 400)
y_true = f(x)

# the number of RBF centers 
n_centers = 8
centers = np.linspace(0, 10, n_centers)

# RBF width
gamma = 0.8

# compute RBF basis functions
Phi = np.zeros((len(x), n_centers))

for i, c in enumerate(centers):
    Phi[:, i] = np.exp(-gamma * (x - c)**2)

# fit linear weights to approximate the function
# solve Phi w = y
w = np.linalg.lstsq(Phi, y_true, rcond=None)[0]

# RBF approximation
y_pred = Phi @ w

# plot
plt.figure(figsize=(12,6))

# true function
plt.plot(x, y_true, linewidth=4, label="True function")

# RBF approximation
plt.plot(x, y_pred, '--', linewidth=3, label="RBF approximation")

# plot RBF splines
colors = plt.cm.rainbow(np.linspace(0,1,n_centers))
for i in range(n_centers):
    plt.fill_between(x, -1, Phi[:,i]-1, color=colors[i], alpha=0.35)

# plot centers
plt.scatter(centers, np.interp(centers, x, y_true),
            color="black", s=120, zorder=5, label="RBF centers")

plt.xlabel("x")
plt.title("Function approximation with RBF features")
plt.legend()
plt.show()