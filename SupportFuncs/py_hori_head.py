import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3d

dt = np.dtype("i4,7198285f4")
X = np.fromfile("HR3_x_sdss3_00.dat", dtype=dt)
Y = np.fromfile("HR3_y_sdss3_00.dat", dtype=dt)
Z = np.fromfile("HR3_z_sdss3_00.dat", dtype=dt)
Vx = np.fromfile("HR3_vx_sdss3_00.dat", dtype=dt)
Vy = np.fromfile("HR3_vy_sdss3_00.dat", dtype=dt)
Vz = np.fromfile("HR3_vz_sdss3_00.dat", dtype=dt)

X = X[0][1]
Y = Y[0][1]
Z = Z[0][1]
Vx = Vx[0][1]
Vy = Vy[0][1]
Vz = Vz[0][1]

R = np.sqrt( X**2 + Y**2 + Z**2 )
V_bulk = np.sqrt( Vx**2 + Vy**2 + Vz**2 )
#print len(R)
#print len(R[R<1000])

print np.min(Vx),np.max(Vx)
print np.min(Vy),np.max(Vy)
print np.min(Vz),np.max(Vz)
print np.min(V_bulk), np.max(V_bulk)

r_cut = 1000

plt.figure()
plt.hist(V_bulk,bins=100)
plt.figure()
plt.hist(Vx,bins=100)
plt.figure()
plt.hist(Vy,bins=100)
plt.figure()
plt.hist(Vz,bins=100)

print "Bulk:"
print np.mean(V_bulk)
print np.std(V_bulk)

X = X[R<r_cut]
Y = Y[R<r_cut]
Z = Z[R<r_cut]
Vx = Vx[R<r_cut]
Vy = Vy[R<r_cut]
Vz = Vz[R<r_cut]

Data = np.array([X,Y,Z,Vx,Vy,Vz])
print np.shape(Data)
np.savetxt("HR3_SDSS_galaxies_0.dat",Data.T,fmt=("%1.4f", "%1.4f", "%1.4f","%1.4f", "%1.4f", "%1.4f"))
#plt.figure()
#plt.plot(X,Y,',')

#plt.figure()
#plt.plot(R[R<r_cut],V_bulk[R<r_cut],',')

#plt.show()

'''
fig = plt.figure(figsize=(10,9))
ax = p3d.Axes3D(fig)
ax.scatter3D(X[0][1],Y[0][1],Z[0][1],s=5)
#ax.set_xlim3d(-1.5,1.5)
#ax.set_ylim3d(-1.5,1.5)
#ax.set_zlim3d(-1.5,1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
'''

'''
lim = -1750
print len(X[0][1][(X[0][1]<lim)] )
plt.figure()
plt.title("Limit x <" + str(lim) + " Mpc/h")
plt.xlabel("Y - Mpc/h")
plt.ylabel("Y - Mpc/h")
plt.plot(Y[0][1][(X[0][1]<lim)],Z[0][1][(X[0][1]<lim)],'.')
plt.show()
'''