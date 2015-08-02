from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
'''
This program reads all the FoF files from HR2 and checks what the range is.
'''

root_dir = "/Users/perandersen/Downloads/"

x = 1800
y = 1800
z = 1800

for i in np.arange(100):
 if i < 10:
     Data = np.genfromtxt(root_dir + "AFoF_halo_cat.00801.0000" + str(i))
 if i >= 10:
     Data = np.genfromtxt(root_dir + "AFoF_halo_cat.00801.000" + str(i))
 Data[:,1] -= x
 Data[:,2] -= y
 Data[:,3] -= z
 Radius = np.sqrt(Data[:,1]**2 + Data[:,2]**2 + Data[:,3]**2)
 
 print np.min(Data[:,1]), np.max(Data[:,1])
 print np.min(Data[:,2]), np.max(Data[:,2])
 print np.min(Data[:,3]), np.max(Data[:,3])
 print ""

