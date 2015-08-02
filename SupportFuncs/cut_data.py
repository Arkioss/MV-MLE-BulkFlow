from __future__ import division
import numpy as np

Data = np.genfromtxt('SamplingLimitNew/Data/SHXYZV1.dat')

Radius = np.sqrt(Data[:,0]**2 + Data[:,1]**2 + Data[:,2]**2)

X = Data[:,0][Radius < 320]
Y = Data[:,1][Radius < 320]
Z = Data[:,2][Radius < 320]
print np.shape(Data)
print np.shape(X)

Data_out = np.zeros((len(X),6))
Data_out[:,0] = X
Data_out[:,1] = Y
Data_out[:,2] = Z
Data_out[:,3] = Data[:,3][Radius < 320]
Data_out[:,4] = Data[:,4][Radius < 320]
Data_out[:,5] = Data[:,5][Radius < 320]
print np.shape(Data_out)
np.savetxt('SamplingLimitNew/Data/Hori320.dat',Data_out,fmt = '%1.4f')