from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as ti

t0 = ti.time()
root_dir = "/Users/perandersen/Data/BulkFlow/DataCommon/"

Halo = np.genfromtxt(root_dir + "Hori1000_angle_10.dat")

#Mock = np.genfromtxt(root_dir + "HR2_mock_0.dat")

print np.shape(Halo)

#print np.shape(Mock)

X_halo = Halo[:,0]
Y_halo = Halo[:,1]
Z_halo = Halo[:,2]

print np.min(X_halo), np.max(X_halo)
print np.min(Y_halo), np.max(Y_halo)
print np.min(Z_halo), np.max(Z_halo)
#X_mock = Mock[:,0]
#Y_mock = Mock[:,1]
#Z_mock = Mock[:,2]

print np.mean(X_halo)
print np.mean(Y_halo)
plt.figure()
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X_halo,Y_halo,'b,')
plt.figure()
plt.xlabel("X")
plt.ylabel("Z")
plt.plot(X_halo,Z_halo,'b,')
plt.figure()
plt.xlabel("Y")
plt.ylabel("Z")
plt.plot(Y_halo,Z_halo,'b,')

'''
plt.figure()
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(X_mock,Y_mock,'r,')
plt.figure()
plt.xlabel("X")
plt.ylabel("Z")
plt.plot(X_mock,Z_mock,'r,')
'''
print "In time ", ti.time() - t0, "seconds"
plt.show()