from __future__ import division
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

'''
Author: Per Andersen, Dark Cosmology Centre
Email: perandersen@dark-cosmology.dk
Last revision: April 2015

'''
def const_volume():
    #1) Spread out point on range -320,320

    X = rnd.uniform(-1000,1000,50000000)
    Y = rnd.uniform(-1000,1000,50000000)
    Z = rnd.uniform(-1000,1000,50000000)

    #2) Define list of angles and other parameters

    #!!! EDIT HERE ONLY !!!
    volume = 40e6 # (Mpc/h)^3 This determines radius of each slice
    Angle = np.array([np.pi , np.pi/2.0, np.pi/4.0, np.pi/8.0])
    #Angle = np.array([0.75*np.pi,0.37*np.pi,0.17*np.pi])
    #Angle = np.array([np.pi/4.])
    bins_x, bins_y, bins_z = 15, 15, 15
    root_dir = "/Users/perandersen/Data/"
    #root_dir = "/home/per/Data/"
    sub_dir = "8A"
    #!!! EDIT HERE ONLY !!!

    #3) Convert to spherical coordinates

    R = np.sqrt(X**2 + Y**2 + Z**2)

    #4) Loop over angles, slice and create histogram

    print "Check number of datapoints for each sample, they should be similar in size!\n"
    for i in np.arange(len(Angle)):
        print "i: ", i
    
        #r_cut = 690.
        r_cut = ( 3*volume / (2*np.pi*(1.0-np.cos(Angle[i]))))**(1./3.)
        print "r: ", r_cut
    
        #Cutting out points beyond wanted radius
        X_cut = X[R<r_cut]
        Y_cut = Y[R<r_cut]
        Z_cut = Z[R<r_cut]
    
    
        #Cutting out points beyond wanted declination
        R_cut = np.sqrt(X_cut**2 + Y_cut**2 + Z_cut**2)
        Theta_cut = np.arccos(Z_cut/R_cut)
    
        X_cut = X_cut[Theta_cut<Angle[i]]
        Y_cut = Y_cut[Theta_cut<Angle[i]]
        Z_cut = Z_cut[Theta_cut<Angle[i]]
        
        #Defining array used to create histogram
        Data = np.array([X_cut,Y_cut,Z_cut])
        Data = Data.T
        print "Datapoints:", np.shape(Data)
        print ""
    
        #Creating histogram
        Hist, Edges = np.histogramdd(Data,bins=(bins_x,bins_y,bins_z),range=((-1000,1000),(-1000,1000),(-1000,1000) ) )
    
        #Testing histogram output
        #plt.figure()
        #plt.xlim((-1000,1000))
        #plt.ylim((-1000,1000))
        #plt.plot(Y_cut,Z_cut,',')
        #plt.figure()
        #plt.ylim((0,45000))
        #for j in np.arange(len(Hist[0,0,:])):
        #    plt.plot(j,np.sum(Hist[j,:,:]),"xb",)
        
        R_cut = np.sqrt(X_cut**2 + Y_cut**2 + Z_cut**2)
        plt.figure()
        plt.hist(R_cut)
        
        plt.show()
    
        Hist = Hist / np.max(Hist)
    
        #Turning 3d histogram into 1d structure
        Hist_to_file = []
        for j in np.arange(np.shape(Hist)[0]):
            for k in np.arange(np.shape(Hist)[1]):
                for l in np.arange(np.shape(Hist)[2]):
                    Hist_to_file.append( Hist[j,k,l] )
    
        #Saving results to files
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Cosmic_hist_" + str(np.round(Angle[i] / np.pi,3)) + ".txt",Hist_to_file,fmt = '%.8f')
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Cosmic_hist_" + str(np.round(Angle[i] / np.pi,3)) + "_x.txt",Edges[0],fmt = '%.8f')
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Cosmic_hist_" + str(np.round(Angle[i] / np.pi,3)) + "_y.txt",Edges[1],fmt = '%.8f')
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Cosmic_hist_" + str(np.round(Angle[i] / np.pi,3)) + "_z.txt",Edges[2],fmt = '%.8f')    

def shells():
    #1) Spread out point on range -320,320

    X = rnd.uniform(-1000,1000,50000000)
    Y = rnd.uniform(-1000,1000,50000000)
    Z = rnd.uniform(-1000,1000,50000000)

    #2) Define listsand other parameters

    #!!! EDIT HERE ONLY !!!
    volume = 150e6 # (Mpc/h)^3 This determines radius of each slice
    R_outer = np.array([330,450,520,600,750,950])
    R_inner = (R_outer**3 - 3.*volume / (4.*np.pi))**(1./3.)
    bins_x, bins_y, bins_z = 12, 12, 12
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "4"
    #!!! EDIT HERE ONLY !!!

    #3) Convert to spherical coordinates

    R = np.sqrt(X**2 + Y**2 + Z**2)

    #4) Loop over angles, slice and create histogram

    print "Check number of datapoints for each sample, they should be similar in size!\n"
    for i in np.arange(len(R_outer)):
        print "i: ", i
    
        print "r: ", R_outer[i]
    
        #Cutting out points beyond wanted radius
        X_cut = X[(R<R_outer[i]) & (R>R_inner[i])]
        Y_cut = Y[(R<R_outer[i]) & (R>R_inner[i])]
        Z_cut = Z[(R<R_outer[i]) & (R>R_inner[i])]
        
        #Defining array used to create histogram
        Data = np.array([X_cut,Y_cut,Z_cut])
        Data = Data.T
        print "Datapoints:", np.shape(Data)
        print ""
    
        #Creating histogram
        Hist, Edges = np.histogramdd(Data,bins=(bins_x,bins_y,bins_z),range=((-1000,1000),(-1000,1000),(-1000,1000) ) )
    
        #Testing histogram output
        #plt.figure()
        #plt.xlim((-1000,1000))
        #plt.ylim((-1000,1000))
        #plt.plot(Y_cut,Z_cut,',')
        #plt.figure()
        #plt.ylim((0,45000))
        #for j in np.arange(len(Hist[0,0,:])):
        #    plt.plot(j,np.sum(Hist[j,:,:]),"xb",)
        
        R_cut = np.sqrt(X_cut**2 + Y_cut**2 + Z_cut**2)
        plt.figure()
        plt.hist(R_cut)
        
        plt.show()
    
        Hist = Hist / np.max(Hist)
    
        #Turning 3d histogram into 1d structure
        Hist_to_file = []
        for j in np.arange(np.shape(Hist)[0]):
            for k in np.arange(np.shape(Hist)[1]):
                for l in np.arange(np.shape(Hist)[2]):
                    Hist_to_file.append( Hist[j,k,l] )
    
        #Saving results to files
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Shell_hist_" + str(R_outer[i]) + ".txt",Hist_to_file,fmt = '%.8f')
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Shell_hist_" + str(R_outer[i]) + "_x.txt",Edges[0],fmt = '%.8f')
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Shell_hist_" + str(R_outer[i]) + "_y.txt",Edges[1],fmt = '%.8f')
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Shell_hist_" + str(R_outer[i]) + "_z.txt",Edges[2],fmt = '%.8f')    

def spheres():
    #1) Spread out point on range -320,320

    X = rnd.uniform(-1000,1000,50000000)
    Y = rnd.uniform(-1000,1000,50000000)
    Z = rnd.uniform(-1000,1000,50000000)

    #2) Define listsand other parameters

    #!!! EDIT HERE ONLY !!!
    #volume = 150e6 # (Mpc/h)^3 This determines radius of each slice
    volume = 40e6 # (Mpc/h)^3 This determines radius of each slice
    R_outer = np.array([330,450,520,600,750,950])
    bins_x, bins_y, bins_z = 12, 12, 12
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "6"
    #!!! EDIT HERE ONLY !!!

    #3) Convert to spherical coordinates

    R = np.sqrt(X**2 + Y**2 + Z**2)

    #4) Loop over angles, slice and create histogram

    print "Check number of datapoints for each sample, they should be similar in size!\n"
    for i in np.arange(len(R_outer)):
        print "i: ", i
    
        print "r: ", R_outer[i]
    
        #Cutting out points beyond wanted radius
        X_cut = X[R<R_outer[i]]
        Y_cut = Y[R<R_outer[i]]
        Z_cut = Z[R<R_outer[i]]
        
        #Defining array used to create histogram
        Data = np.array([X_cut,Y_cut,Z_cut])
        Data = Data.T
        print "Datapoints:", np.shape(Data)
        print ""
    
        #Creating histogram
        Hist, Edges = np.histogramdd(Data,bins=(bins_x,bins_y,bins_z),range=((-1000,1000),(-1000,1000),(-1000,1000) ) )
    
        #Testing histogram output
        #plt.figure()
        #plt.xlim((-1000,1000))
        #plt.ylim((-1000,1000))
        #plt.plot(Y_cut,Z_cut,',')
        #plt.figure()
        #plt.ylim((0,45000))
        #for j in np.arange(len(Hist[0,0,:])):
        #    plt.plot(j,np.sum(Hist[j,:,:]),"xb",)
        
        R_cut = np.sqrt(X_cut**2 + Y_cut**2 + Z_cut**2)
        plt.figure()
        plt.hist(R_cut)
        
        plt.show()
    
        Hist = Hist / np.max(Hist)
    
        #Turning 3d histogram into 1d structure
        Hist_to_file = []
        for j in np.arange(np.shape(Hist)[0]):
            for k in np.arange(np.shape(Hist)[1]):
                for l in np.arange(np.shape(Hist)[2]):
                    Hist_to_file.append( Hist[j,k,l] )
    
        #Saving results to files
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Sphere_hist_" + str(R_outer[i]) + ".txt",Hist_to_file,fmt = '%.8f')
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Sphere_hist_" + str(R_outer[i]) + "_x.txt",Edges[0],fmt = '%.8f')
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Sphere_hist_" + str(R_outer[i]) + "_y.txt",Edges[1],fmt = '%.8f')
        np.savetxt(root_dir + "BulkFlow/" + sub_dir + "/Histogram/Sphere_hist_" + str(R_outer[i]) + "_z.txt",Edges[2],fmt = '%.8f')    

const_volume()
#spheres()
