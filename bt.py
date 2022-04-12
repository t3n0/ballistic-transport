import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from tqdm import tqdm

def collision(r0, v0):
    t1 = (v0[0]*(segs[:,0,1]-r0[1]) - v0[1]*(segs[:,0,0]-r0[0]))/(v0[0]*(segs[:,0,1]-segs[:,1,1]) - v0[1]*(segs[:,0,0]-segs[:,1,0]))
    t2 = ((segs[:,0,1]-segs[:,1,1])*(segs[:,0,0]-r0[0]) - (segs[:,0,1]-r0[1])*(segs[:,0,0]-segs[:,1,0]))/(v0[0]*(segs[:,0,1]-segs[:,1,1]) - v0[1]*(segs[:,0,0]-segs[:,1,0]))
    t3 = np.where((t1<=1.0) * (t1>=0.0) * (t2>0.0) * mask, t2, np.inf)
    idx = np.argmin(t3)
    t = t2[idx]
    return t, idx

# a = np.random.random((16, 16))
# plt.imshow(a, cmap='coolwarm', interpolation='spline16')
# plt.show()

#micrometers
points = np.array([[0, 0], [0, 4], [1, 4], [3, 1.5], [3, 4], [10, 4], [10, 0], [9,0], [9,3], [7,3], [7,0], [6,0], [6,3],[4,3],[4,0],[3,0],[1,2.5],[1,0]])
segs = []

for i in range(len(points)):
    segs.append([points[i], points[(i+1) % len(points)]])

segs = np.array(segs)

xmin = np.min(points[:,0])
xmax = np.max(points[:,0])
ymin = np.min(points[:,1])
ymax = np.max(points[:,1])

xgrid, dx = np.linspace(xmin, xmax, 200, retstep=True)
ygrid, dy = np.linspace(ymin, ymax, 200, retstep=True)
x, y = np.meshgrid(xgrid, ygrid)
bins = x*0.0 + y*0.0

tau = 10
tmax = 10 #ps

for jj in tqdm(range(1000)):
    r0 = np.array([0.5, 1.5]) + 0.1*(1-2*np.random.random(2))
    v0 = 1*(1-2*np.random.random(2)) #microm/ps

    t = 0
    mask = [True for i in range(len(points))]
    while t < tmax:
        tnew, idx = collision(r0, v0)
        mask = [True if i != idx else False for i in range(len(points))]
        rnew = r0 + v0*tnew
        absv0 = np.linalg.norm(v0)
        rxs = np.arange(r0[0], rnew[0], v0[0]*dx/absv0/5)
        rys = np.arange(r0[1], rnew[1], v0[1]*dx/absv0/5)
        for rx, ry in zip(rxs, rys):
            deltat = np.linalg.norm(r0 - np.array([rx,ry]))/absv0
            bins[(rx>x)*(rx<x+dx) * (ry>y)*(ry<y+dy)] += np.exp(-(t+deltat)/tau)
        pollo = segs[idx,1] - segs[idx,0]
        v0new = 2*np.dot(v0,pollo)/np.dot(pollo,pollo)*pollo - v0
        r0 = rnew
        v0 = v0new
        t += tnew

section = np.array([[0.5,0.0],[0.5,3.25],[1.0,3.25],[3,0.75],[3.5,0.75],[3.5,3.5],[6.5,3.5]])


#plt.imshow(bins, cmap='coolwarm', interpolation='spline16')
plt.pcolormesh(x, y, bins, cmap='coolwarm', vmin=0, vmax=np.max(bins))
plt.plot(segs[:,:,0], segs[:,:,1], 'k')
plt.plot(section[:,0],section[:,1],'r')
plt.gca().set_aspect('equal')
plt.show()

NN = 100
profile = np.zeros(NN*(len(section)-1))
jj = 0
for i in range(len(section)-1):
    rxs = np.linspace(section[i,0],section[i+1,0],NN)
    rys = np.linspace(section[i,1],section[i+1,1],NN)
    #print(rxs)
    for rx, ry in zip(rxs, rys):
        #print(bins[(rx>x)*(rx<x+dx) * (ry>y)*(ry<y+dy)][0])
        try:
            profile[jj] = bins[(rx>x)*(rx<x+dx) * (ry>y)*(ry<y+dy)]
        except:
            pass
        jj += 1

profilegrid = np.array([0])
for i in range(len(section)-1):
    length = np.linalg.norm(section[i+1]-section[i])
    aux = np.linspace(0, length, NN)
    profilegrid = np.concatenate((profilegrid, aux+profilegrid[-1]))
#print(profile)
plt.plot(profilegrid[1:], profile)
plt.show()
