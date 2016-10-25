from __future__ import division
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt


#0) Define constants
volume = 150000000 # (Mpc/h)^3

#1) Spread out point on range -320,320

X = rnd.uniform(-1000,1000,10000000)
Y = rnd.uniform(-1000,1000,10000000)
Z = rnd.uniform(-1000,1000,10000000)

#2) Define list of angles

Angle = np.array([np.pi , np.pi/2.0, np.pi/4.0, np.pi/8.0])

#3) Convert to spherical coordinates

R = np.sqrt(X**2 + Y**2 + Z**2)
Theta = np.arccos(Z/R)
Phi = np.arctan2(Y,X) + np.pi
Dec = 90.0 - 180.0 * Theta / np.pi
Ra = 180.0 * Phi / np.pi
print ""
print np.min(R), np.max(R)
print np.min(Theta), np.max(Theta)
print np.min(Phi), np.max(Phi)
print ""
print np.min(Dec), np.max(Dec)
print np.min(Ra), np.max(Ra)
print ""

#4) Loop over angles, slice and create histogram

for i in np.arange(len(Angle)):
    r_cut = ( 3*volume / (2*np.pi*(1.0-np.cos(Angle[i]))))**(1.0/3.0)
    print "r: ", r_cut
    
    R_cut = R[R<r_cut]
    Theta_cut = Theta[R<r_cut]
    Phi_cut = Phi[R<r_cut]
    
    R_cut = R_cut[Theta_cut<Angle[i]]
    Theta_cut = Theta_cut[Theta_cut<Angle[i]]
    Phi_cut = Phi_cut[Theta_cut<Angle[i]]
    
    Dec = 90.0 - 180.0 * Theta_cut / np.pi
    Ra = 180.0 * Phi_cut / np.pi
    
    Data = np.array([Ra,Dec,R_cut])
    Data = Data.T
    print np.shape(Data)
    
    Hist, Edges = np.histogramdd(Data,bins=(16,16,16),range=((0,360),(-90,90),(0,1000) ) )
    W = np.zeros((16,16,16))
    for j in np.arange(16):
      for k in np.arange(16):
        for l in np.arange(16):
          W[j,k,l] = (  Edges[2][l+1]**3 - Edges[2][l]**3  ) * (  np.cos(Edges[1][k]) - np.cos(Edges[1][k+1])  )
          print j,k,l,W[j,k,l]
    W = W / np.min(W)
    Hist = Hist / W
    Hist = Hist / np.max(Hist)
    Hist_to_file = []
    for j in np.arange(np.shape(Hist)[0]):
        for k in np.arange(np.shape(Hist)[1]):
            for l in np.arange(np.shape(Hist)[2]):
                Hist_to_file.append( Hist[j,k,l] )
    print "i: ", i
    np.savetxt("/Users/perandersen/Data/DataCommon/Angle_hist_" + str(np.round(Angle[i] / np.pi,3)) + ".txt",Hist_to_file,fmt = '%.8f')
    np.savetxt("/Users/perandersen/Data/DataCommon/Angle_hist_" + str(np.round(Angle[i] / np.pi,3)) + "_ra.txt",Edges[0],fmt = '%.8f')
    np.savetxt("/Users/perandersen/Data/DataCommon/Angle_hist_" + str(np.round(Angle[i] / np.pi,3)) + "_dec.txt",Edges[1],fmt = '%.8f')
    np.savetxt("/Users/perandersen/Data/DataCommon/Angle_hist_" + str(np.round(Angle[i] / np.pi,3)) + "_mpch.txt",Edges[2],fmt = '%.8f')    
    