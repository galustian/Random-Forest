import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Currently only works with Tree-based models (can't interpret one-hot vectors)
def plot_decision_region(X, Y, model=None, cmap='rainbow', alpha=0.3, s=20.0):
    if model == None:
        raise ValueError('model must be specified')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # X_boundary, Y_boundary
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

    """
    np.c_ Example
    --------
    >>> np.c_[np.array([1,2,3]), np.array([4,5,6])]
    array([[1, 4],
            [2, 5],
            [3, 6]])
    """
    # .ravel() unpacks the array ([[1, 2, 3], [1, 2, 3]] => [1, 2, 3, 1, 2, 3])
    XY_2D_datapoints = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(XY_2D_datapoints)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=cm.get_cmap(cmap), alpha=alpha, antialiased=True)
    plt.axis('off') # turns off lines and labels

    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm.get_cmap(cmap), s=s, edgecolor='black', linewidths=0.5, marker='o')
    plt.show()