#
# Copyright (c) 2022 Stefano Dal Forno.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

def segments(points):
    segs = []
    for i in range(len(points)):
        segs.append([points[i], points[(i+1) % len(points)]])
    return np.array(segs)

def winding(point, polygon):
    angle = 0
    for i in range(len(polygon)):
        aux1 = polygon[i]-point
        aux2 = polygon[(i+1) % len(polygon)]-point
        newangle = np.arctan2(aux1[0]*aux2[1]-aux1[1]*aux2[0], aux1[0]*aux2[0]+aux1[1]*aux2[1])
        angle += newangle
    return not -0.1<angle/2/np.pi<0.1

def r0init(MC):
    r0s = np.zeros((MC,2))
    ii = 0
    while ii < MC:
        r0 = np.random.multivariate_normal(r0mean, [[r0var,0], [0,r0var]])
        if winding(r0, points):
            r0s[ii] = r0
            ii +=1
    return r0s

def v0init(MC):
    theta = 2*np.pi*np.random.random(MC)
    v0norm = np.random.normal(v0mean, v0var, MC)
    v0s = np.ones((MC,2))
    v0s[:,0] = v0norm*np.cos(theta)
    v0s[:,1] = v0norm*np.sin(theta)
    return v0s

# physical parameters
tau = 20                    # ps
tmax = 100                  # ~5*tau
r0mean = [1, 3]             # micrometers
r0var = 0.8
v0mean = 10.0                # micron/ps
v0var = 0.1
phi = 0.1*np.pi
MC = 10000

filename = f'MC{MC}tau{tau:.0f}phi{phi:.2f}vel{v0mean:.1f}.pdf'
points = np.array([[0, 0], [0, 10], [2, 10], [8, 3], [8, 10], [20, 10], [20, 0], [18,0], [18,8], [15,8], [15,0], [13,0], [13,8],[10,8],[10,0],[8,0],[2,7],[2,0]])
section = np.array([[1,0],[1,8.5],[2,8.5],[8,1.5],[9,1.5],[9,9],[20,9]])
segs = segments(points)

edge = 0.5
xmin = np.min(points[:,0]) - edge
xmax = np.max(points[:,0]) + edge
ymin = np.min(points[:,1]) - edge
ymax = np.max(points[:,1]) + edge

xsize, ysize = 400, 200
xgrid, dx = np.linspace(xmin, xmax, xsize, retstep=True)
ygrid, dy = np.linspace(ymin, ymax, ysize, retstep=True)
bins = np.zeros((xsize-1,ysize-1))

r0s = r0init(MC)
v0s = v0init(MC)

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
fig.set_size_inches(7, 5)
axs[0].set_position([0.37, 0.48, 0.65, 0.88])
axs[1].set_position([0.125, 0.11, 0.9, 0.46])
axs[0].imshow(bins.T, cmap='coolwarm', origin='lower', extent=[xmin, xmax, ymin, ymax], interpolation='bicubic')
axs[0].plot(segs[:,:,0], segs[:,:,1], 'k')
axs[0].plot(section[:,0],section[:,1],'r')
axs[0].set_xlim(xmin, xmax)
axs[0].set_aspect('equal')
axs[0].set_xlabel('x (μm)')
axs[0].set_ylabel('y (μm)')

axs[1].plot(profilegrid, profile/max(profile),'k')
axs[1].axhline(0, linestyle='--', color='grey')
for vl in vline:
    axs[1].axvline(x=vl,color='r')
axs[1].set_xlabel('Distance (μm)')
axs[1].set_ylabel('Intensity (a.u.)')

text = f'MC steps = {MC}\ntmax = {tmax:.1f} ps\ntau = {tau:.1f} ps\nr0 = ({r0mean[0]:.1f},{r0mean[1]:.1f}) ± {r0var} μm\nv0 = {v0mean:.1f} ± {v0var} μm/ps\ndiffusion = ±{phi:.2f} rads'
plt.gcf().text(0.05,1.0,text)

plt.savefig(filename, bbox_inches='tight')
