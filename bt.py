import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def collision(r0, v0):
    D = (v0[0]*(segs[:,0,1]-segs[:,1,1]) - v0[1]*(segs[:,0,0]-segs[:,1,0]))
    t1 = (v0[0]*(segs[:,0,1]-r0[1]) - v0[1]*(segs[:,0,0]-r0[0]))/D
    t2 = ((segs[:,0,1]-segs[:,1,1])*(segs[:,0,0]-r0[0]) - (segs[:,0,1]-r0[1])*(segs[:,0,0]-segs[:,1,0]))/D
    t3 = np.where((t1<=1.0) * (t1>=0.0) * (t2>0.0) * mask, t2, np.inf)
    idx = np.argmin(t3)
    t = t2[idx]
    return t, idx

def trajectory(r0,r1,t,tnew):
    r1n = np.array([int((r1[0]-xmin)/dx), int((r1[1]-ymin)/dy)])
    r0n = np.array([int((r0[0]-xmin)/dx), int((r0[1]-ymin)/dy)])
    N = abs(r1n[0]-r0n[0]) + 1
    M = abs(r1n[1]-r0n[1]) + 1
    C = N + M - np.gcd(N,M)
    xsteps = np.rint(np.linspace(r0n[0],r1n[0],C)).astype(int)
    ysteps = np.rint(np.linspace(r0n[1],r1n[1],C)).astype(int)
    tsteps = np.linspace(t,t+tnew,C)
    return [(i,j,t) for i,j,t in zip(xsteps, ysteps, tsteps)]

def scattering(phi):
    r12 = segs[idx,1] - segs[idx,0]
    vpara = np.dot(v0,r12)/np.dot(r12,r12)*r12
    if phi == 0.0:
        return 2*vpara - v0
    else:
        absv0 = np.linalg.norm(v0)
        vperp = v0 - vpara
        theta = np.arccos(np.dot(r12,v0)/absv0/np.linalg.norm(r12))
        alphaplus = np.cos(max(theta-phi, 0))/np.cos(theta)
        alphaminus = np.cos(min(theta+phi,np.pi))/np.cos(theta)
        absvpara = np.linalg.norm(vpara)
        absvperp = np.linalg.norm(vperp)
        alpha = (alphaplus-alphaminus)*np.random.random() + alphaminus
        beta = np.sqrt(1 + absvpara**2/absvperp**2*(1-alpha**2))
        return alpha*vpara - beta*vperp

# physical parameters
tau = 20 #ps
tmax = 100 #ps #5*tau

#micrometers
points = np.array([[0, 0], [0, 10], [2, 10], [8, 3], [8, 10], [20, 10], [20, 0], [18,0], [18,8], [15,8], [15,0], [13,0], [13,8],[10,8],[10,0],[8,0],[2,7],[2,0]])
section = np.array([[1,0],[1,8.5],[2,8.5],[8,1.5],[9,1.5],[9,9],[20,9]])
segs = []

for i in range(len(points)):
    segs.append([points[i], points[(i+1) % len(points)]])

segs = np.array(segs)

edge = 0.5
xmin = np.min(points[:,0]) - edge
xmax = np.max(points[:,0]) + edge
ymin = np.min(points[:,1]) - edge
ymax = np.max(points[:,1]) + edge

xsize, ysize = 400, 200
xgrid, dx = np.linspace(xmin, xmax, xsize, retstep=True)
ygrid, dy = np.linspace(ymin, ymax, ysize, retstep=True)
bins = np.zeros((xsize-1,ysize-1))

MC = 100
r0mean = [1, 3]
r0var = [[0.8,0], [0,0.8]]

r0s = np.random.multivariate_normal(r0mean, r0var, MC)
r0s = r0s[(r0s[:,0]<2) * (r0s[:,0]>0) * (r0s[:,1]>0) * (r0s[:,1]<10)]
MC = len(r0s)

theta = 2*np.pi*np.random.random(MC)
v0norm = np.random.normal(1.0, 0.1, MC)
v0s = np.ones((MC,2))
v0s[:,0] = v0norm*np.cos(theta)
v0s[:,1] = v0norm*np.sin(theta)

phi = 0.1*np.pi
for jj in tqdm(range(MC)):
    r0 = r0s[jj]
    v0 = v0s[jj]

    # r0 = np.array([1,3])
    # v0 = np.array([0.68,0.72])

    t = 0
    mask = [True for i in range(len(points))]
    while t < tmax:
        tnew, idx = collision(r0, v0)
        mask = [True if i != idx else False for i in range(len(points))]
        r1 = r0 + v0*tnew
        traj = trajectory(r0,r1,t,tnew)
        for n,m,dt in traj:
            bins[n,m] += np.exp(-dt/tau)
        r0 = r1
        r12 = segs[idx,1] - segs[idx,0]
        v0 = scattering(phi)
        t += tnew

profile = []
profilegrid = []
last = 0
vline = [0]
for i in range(len(section)-1):
    delta = np.linalg.norm(section[i+1] - section[i])
    traj = trajectory(section[i], section[i+1], last, delta)
    for n,m,d in traj:
        profile.append(bins[n,m])
        profilegrid.append(d)
    last += delta
    vline.append(last)
profile = np.array(profile)
profilegrid = np.array(profilegrid)

fig, axs = plt.subplots(2)
axs[0].imshow(bins.T, cmap='coolwarm', origin='lower', extent=[xmin, xmax, ymin, ymax], interpolation='bicubic')
axs[0].plot(segs[:,:,0], segs[:,:,1], 'k')
axs[0].plot(section[:,0],section[:,1],'r')
axs[0].set_aspect('equal')

axs[1].plot(profilegrid, profile)
for vl in vline:
    axs[1].axvline(x=vl,color='r')

plt.show()
