from cmath import inf
import numpy as np
from numba import njit
import time

points = [[1, 1], [1, 2], [2, 3], [3, 3], [3, 1], [2, 2]]
segs = []

for i in range(len(points)):
    segs.append([points[i], points[(i+1) % len(points)]])

segs = np.array(segs)
r0 = np.array([3/2, 2])
v0 = np.array([1.01, 1])

def _collision(r0, v0):
    t1 = (v0[0]*(segs[:,0,1]-r0[1]) - v0[1]*(segs[:,0,0]-r0[0]))/(v0[0]*(segs[:,0,1]-segs[:,1,1]) - v0[1]*(segs[:,0,0]-segs[:,1,0]))
    t2 = ((segs[:,0,1]-segs[:,1,1])*(segs[:,0,0]-r0[0]) - (segs[:,0,1]-r0[1])*(segs[:,0,0]-segs[:,1,0]))/(v0[0]*(segs[:,0,1]-segs[:,1,1]) - v0[1]*(segs[:,0,0]-segs[:,1,0]))
    t3 = np.where((t1<=1.0) * (t1>=0.0) * (t2>0.0), t2, np.inf)
    idx = np.argmin(t3)
    t = t2[idx]
    r0new = r0 + v0*t
    pollo = segs[idx,1] - segs[idx,0]
    v0new = 2*np.dot(v0,pollo)/np.dot(pollo,pollo)*pollo - v0
    return t, r0new, v0new

timeX = njit()(_collision)

start = time.process_time()
for i in range(10):
    t, r0, v0 = _collision(r0, v0)
    print(t, r0, v0)
print(time.process_time()-start)
