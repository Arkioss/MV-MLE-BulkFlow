from __future__ import division
import numpy as np

'''
Author: Per Andersen, Dark Cosmology Centre
Email: perandersen@dark-cosmology.dk
Last revision: April 2015

'''

#Number of datasets to find length for. Look at /Users/perandersen/Data/DataCosmic/ . 
n = 100
root_dir = "/Users/perandersen/Data/"
#root_dir = "/home/per/Data/"

#Reading datasets, getting length and appending to list
Len = []
print np.arange(n)

for i in np.arange(n): 
 print "i: ", i
 Data = np.genfromtxt(root_dir + "BulkFlow/DataCommon/Hori1000_angle_" + str(i) + ".dat")
 Len.append(len(Data))

#Writing list to file
print np.min(Len), np.max(Len) #Output to check file was properly read
np.savetxt(root_dir + "BulkFlow/DataCommon/Hori_len.txt",Len,fmt=("%6d"))
