import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

coords = np.array([[0,0], [0,1], [1,0], [1,1]])
tri = Delaunay(coords)

print(tri.simplices)

plt.triplot(coords[:,0], coords[:,1], tri.simplices, 'o-')
plt.show()
