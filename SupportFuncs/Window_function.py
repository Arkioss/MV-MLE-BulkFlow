from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time as ti

print "Beginning..."
t0 = ti.time()

#For interpolating redshifts to distances
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

def legend_plot(location=1,fsize=18,ncols=1):
 leg = plt.legend(loc=location,ncol=ncols)
 leg.draw_frame(False)
 for x in leg.get_texts():
  x.set_fontsize(fsize)


def window_1():

    Redshift_interp = np.genfromtxt("interp_redshift_h.txt",dtype=np.float128)
    Lum_dist_interp = np.genfromtxt("interp_lum_dist_h.txt",dtype=np.float128)

    #Importing data from morags code
    root_dir = "/Users/perandersen/Dropbox/BulkFlow/Morag/"
    Weights = np.genfromtxt(root_dir + "Output/weights_RI50_mock_100_RI10.dat",dtype=np.float128)
    Data = np.genfromtxt(root_dir + "Data/mock_100_RI10.dat",dtype=np.float128)
    Power = np.genfromtxt(root_dir + "Data/pk_planck_z0_nonlinear_matterpower.dat")

    K = Power[:,0]
    Pk = Power[:,1]

    Wx = Weights[:,0]
    Wy = Weights[:,1]
    Wz = Weights[:,2]

    Ra = Data[:,0]
    Dec = Data[:,1]
    Redshift = Data[:,2]

    #Converting redshifts to radii
    Radius = np.zeros(len(Redshift))
    for i in np.arange(len(Radius)):
        Radius[i] = lin_interp(Redshift_interp,Lum_dist_interp,Redshift[i])

    #Converting to cartesian
    Theta = (90. - Dec) * np.pi / 180.
    Phi = Ra * np.pi / 180.

    X_hat = np.sin(Theta) * np.cos(Phi)
    Y_hat = np.sin(Theta) * np.sin(Phi)
    Z_hat = np.cos(Theta)

    weights_size = len(Weights)
    k_size = len(K)

    f = np.zeros((weights_size,weights_size,k_size))

    for m in np.arange(weights_size):
      print (m / weights_size) * 100., "% Done"
      for n in np.arange(m+1):
       if (m == n):
         f[m,n,:] = 1./3.
       else:
         for i in np.arange(k_size):
           cosalpha = X_hat[m] * X_hat[n] + Y_hat[m] * Y_hat[n] + Z_hat[m] * Z_hat[n]
           sin2alpha = 1. - cosalpha**2
           val = Radius[m]**2 + Radius[n]**2 - 2 * Radius[m] * Radius[n] * cosalpha
           if (val < 0):
             print "Negative val!"
           area = np.sqrt(  val  )
           area2 = area*area
           kA = K[i] * area
           kA2 = kA*kA
           sinkA = np.sin(kA)
           coskA = np.cos(kA)
       
           c1 = cosalpha / 3.
           c2 = Radius[n] * Radius[m] * sin2alpha / area2
           #fmnk = c1*(sinKA/kA*(3.-6./kA2)+6.*cosKA/kA2)  + c2* ((3./kA2-1.)*sinKA/kA - 3.*cosKA/kA2);
           f[m,n,i] = c1 * (  sinkA*(3. - 6./kA2) / kA + 6.*coskA/kA2  ) + c2 * ( (3./kA2 - 1.) * sinkA / kA - 3.*coskA/kA2   )

    for m in np.arange(weights_size):
      for n in np.arange(m+1,weights_size):
        f[m,n,:] = f[n,m,:]

    Window_func_x = np.zeros(k_size)
    Window_func_y = np.zeros(k_size)
    Window_func_z = np.zeros(k_size)

    for i in np.arange(k_size):
      for m in np.arange(weights_size):
        for n in np.arange(weights_size):
          Window_func_x[i] += Wx[n] * Wx[m] * f[m,n,i]
          Window_func_y[i] += Wy[n] * Wy[m] * f[m,n,i]
          Window_func_z[i] += Wz[n] * Wz[m] * f[m,n,i]

    plt.figure()
    plt.title("W_ii")
    plt.plot(K,Window_func_x,label="W_xx")
    plt.plot(K,Window_func_y,label="W_yy")
    plt.plot(K,Window_func_z,label="W_zz")
    legend_plot()
    plt.xlim((0,0.5))

def window_2():
    root_dir = "/Users/perandersen/Dropbox/BulkFlow/Morag/"

    Data10 = np.genfromtxt(root_dir + "Output/W2_RI10_mock_100_RI10.dat")
    Data50 = np.genfromtxt(root_dir + "Output/W2_RI50_mock_100_RI10.dat")
    Data70 = np.genfromtxt(root_dir + "Output/W2_RI70_mock_100_RI10.dat")

    Wxx10 = np.array([])
    Wyy10 = np.array([])
    Wzz10 = np.array([])
    for i in np.arange(len(Data10)):
     if (Data10[i,0] == 0):
       Wxx10 = np.append(Wxx10,Data10[i,2])
     if (Data10[i,0] == 4):
       Wyy10 = np.append(Wyy10,Data10[i,2])
     if (Data10[i,0] == 8):
       Wzz10 = np.append(Wzz10,Data10[i,2])

    Wxx50 = np.array([])
    Wyy50 = np.array([])
    Wzz50 = np.array([])
    for i in np.arange(len(Data50)):
     if (Data50[i,0] == 0):
       Wxx50 = np.append(Wxx50,Data50[i,2])
     if (Data50[i,0] == 4):
       Wyy50 = np.append(Wyy50,Data50[i,2])
     if (Data50[i,0] == 8):
       Wzz50 = np.append(Wzz50,Data50[i,2])

    K = np.linspace(0.0001,1.9981,len(Wxx10))
    plt.figure()
    plt.xlabel(r'$k$ [$h$ Mpc$^{-1}$]',size='xx-large')
    plt.ylabel(r'$W^2_{ii}(k)$',size='xx-large')
    plt.plot(K,Wxx50,'b--',label=r'$W^2_{xx}$, $R_{I}$ = 50 Mpc $h^{-1}$')
    plt.plot(K,Wyy50,'b-',label=r'$W^2_{yy}$, $R_{I}$ = 50 Mpc $h^{-1}$')
    plt.plot(K,Wzz50,'b-.',label=r'$W^2_{zz}$, $R_{I}$ = 50 Mpc $h^{-1}$')
    plt.plot(K,Wxx10,'r--',label=r'$W^2_{xx}$, $R_{I}$ = 10 Mpc $h^{-1}$')
    plt.plot(K,Wyy10,'r-',label=r'$W^2_{yy}$, $R_{I}$ = 10 Mpc $h^{-1}$')
    plt.plot(K,Wzz10,'r-.',label=r'$W^2_{zz}$, $R_{I}$ = 10 Mpc $h^{-1}$')
    plt.xlim((0,0.5))
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=20)

def power_spectrum():
    root_dir = "/Users/perandersen/Dropbox/BulkFlow/Morag/"
    
    Power = np.genfromtxt(root_dir + "Data/pk_planck_z0_nonlinear_matterpower.dat")

    K = Power[:,0]
    Pk = Power[:,1]
    
    plt.figure()
    plt.xlabel(r'$k$ [$h$ Mpc$^{-1}$]',size='xx-large')
    plt.ylabel(r'$P(k)$',size='xx-large')
    plt.plot(K,Pk,lw=4)
    plt.xlim((0,0.5))
    #plt.xscale('log')
    plt.yscale('log')
    plt.subplots_adjust(bottom=0.12,left=0.14)
window_2()
#power_spectrum()
print "Done in time: ", ti.time() - t0
plt.show()

