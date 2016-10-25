from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#Test = np.genfromtxt("Data/MV_Bulk_flows_0.txt")
Test = np.genfromtxt("DataCosmic/MLE_Bulk_flows_0A.txt")

n = 50
Means = []
Stds = []

Bulks = np.zeros((n,len(Test)))
for i in np.arange(n):
 #Data = np.genfromtxt("Data/MV_Bulk_flows_" + str(i) + ".txt")
 Data = np.genfromtxt("DataCosmic/MLE_Bulk_flows_" + str(i) + "A.txt")
 Bulks[i,:] = Data
 Means.append(np.mean(Data))
 Stds.append(np.std(Data))
 #plt.figure()
 #plt.hist(Bulks[i,:],10)
 #plt.show()
Completeness = Bulks.reshape((np.shape(Bulks)[0]*np.shape(Bulks)[1],1))

#print Stds
#print np.mean(Stds), np.std(Stds)

plt.figure()
plt.xlim((0,500))
plt.hist(Means,range=(0,500),bins=20)

plt.figure()
plt.xlim((0,500))
plt.hist(Completeness,range=(0,500),bins=20)

print "Cosmic Variance"
print "Mean: ", np.mean(Completeness)
print "Std: ", np.std(Completeness)

plt.show()