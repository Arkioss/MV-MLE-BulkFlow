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

root_dir = "/Users/perandersen/Data/"
sub_dir = "0A"

def create_mock(nn,rmax):

    #Creating points in spherical coordinates
    Phi = np.random.uniform(0,2*np.pi,nn)
    Theta = np.arccos(2*np.random.uniform(0,1,nn) - 1)
    R = np.ones(nn)*rmax #np.random.uniform(0,rmax,nn)
    #Here the radial velocity is set
    Vr = 100*np.ones(nn)
    
    
    #Converting to cartesian
    X = R * np.sin(Theta) * np.cos(Phi)
    Y = R * np.sin(Theta) * np.sin(Phi)
    Z = R * np.cos(Theta)
    Vx = Vr * np.sin(Theta) * np.cos(Phi) + 100. #100. * np.ones(nn)
    Vy = Vr * np.sin(Theta) * np.sin(Phi) #100.* np.ones(nn)
    Vz = Vr * np.cos(Theta) #0.* np.ones(nn)
    
    #Checks
    print np.sqrt(Vx**2 + Vy**2 + Vz**2)
    
    Mock = np.array([X,Y,Z,Vx,Vy,Vz]).T
    
    return Mock
    
def bulk_test():
    Bulks = np.array([])
    for i in np.arange(60):
      Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_1.0_" + str(i) + ".txt")
      Bulks = np.append(Bulks,Data)
    print np.mean(Bulks), " +/- ", np.std(Bulks)
    plt.figure()
    plt.hist(Bulks)
    plt.show()

def bulk_test_new():
    Bulks_x = np.array([])
    Bulks_y = np.array([])
    Bulks_z = np.array([])
    
    for i in np.arange(60):
      Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_1.0_" + str(i) + ".txt")
      Bulks_x = np.append(Bulks_x,Data[:,0])
      Bulks_y = np.append(Bulks_y,Data[:,1])
      Bulks_z = np.append(Bulks_z,Data[:,2])
    print np.mean(Bulks_x), " +/- ", np.std(Bulks_x)
    print np.mean(Bulks_y), " +/- ", np.std(Bulks_y)
    print np.mean(Bulks_z), " +/- ", np.std(Bulks_z)
    print np.sqrt(np.mean(Bulks_x)**2 + np.mean(Bulks_y)**2 + np.mean(Bulks_z)**2)
    #plt.figure()
    #plt.hist(Bulks_x)
    #plt.show()
      




#for i in np.arange(60):
#  Mock = create_mock(10000,330)
#  np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Hori_sub_cart_1.0_" + str(i) + ".txt",Mock,fmt='%1.4f')

#bulk_test()
bulk_test_new()