from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def lin_interp(X,Y,x):
 i = 0
 if ( (x > np.max(X)) or (x < np.min(X)) ):
  print "Interpolation Impossible!", exit()
 while (X[i] < x):
  i += 1
 if i == 0:
  return Y[i] + (Y[i+1]-Y[i]) * (x - X[i]) / (X[i+1] - X[i])
 else:
  return Y[i-1] + (Y[i]-Y[i-1]) * (x - X[i-1]) / (X[i] - X[i-1])

Redshift_interp = np.genfromtxt("interp_redshift_h.txt")
Lum_dist_interp = np.genfromtxt("interp_lum_dist_h.txt")

def plot_mock_ideal():
    Ideal = np.genfromtxt("/Users/perandersen/Data/DataCommon/mock_ideal_RI10.dat")
    
    Redshift = Ideal[:,2]
    Ra = Ideal[:,0]
    Dec = Ideal[:,1]
    
    plt.figure()
    plt.xlim((0,360))
    plt.ylim((-90,90))
    plt.plot(Ra,Dec,'x')
    
    plt.figure()
    plt.hist(Ra,10)
    
    plt.figure()
    plt.hist(Dec,10)
    
    Lum_dist = np.zeros(len(Redshift))
    
    for i in np.arange(len(Lum_dist)):
        Lum_dist[i] = lin_interp(Redshift_interp,Lum_dist_interp,Redshift[i])
    
    #plt.figure()
    #plt.title("Redshift")
    #plt.hist(Redshift,30)
    
    plt.figure()
    plt.title("Lum dist")
    plt.hist(Lum_dist,30)

def create_ideal(nn,ri):
    #Defining PDF
    
    R = np.linspace(0,4*ri,1000)
    r_max = 2 * ri * ri * np.exp(-1.0)
    r_func = lambda r: r*r * np.exp(  -r*r / (2*ri*ri)  )
    
    R_out = np.array([])
    Z_out = np.array([])
    
    while(len(R_out) < nn):
      a = np.random.choice(R)
      b = np.random.uniform(0,r_max,1)
      if (b < r_func(a)):
        R_out = np.append(R_out,a)
        Z_out = np.append(Z_out,lin_interp(Lum_dist_interp,Redshift_interp,R_out[-1]))
    
    
    Ra = np.random.uniform(0,360,nn)
    Dec = (np.arccos(2*np.random.uniform(0,1,nn) - 1)) * 180.0 / 3.14159 - 90.0
    print np.min(Ra),np.max(Ra)
    print np.min(Dec),np.max(Dec)
    return Ra, Dec, Z_out
#plot_mock_ideal()

radius_ideal = 500
number_mock = 1200
Ra,Dec,Z = create_ideal(number_mock,radius_ideal)

np.savetxt("mock_ideal_ri" + str(radius_ideal) + "_n" + str(number_mock) + ".txt",np.array([Ra,Dec,Z]).T,fmt='%1.8f')
