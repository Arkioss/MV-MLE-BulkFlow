from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


#---------------------------------- DEFINING FUNCTIONS ----------------------------------
def legend_plot(location=1,fsize=18,ncols=1):
 leg = plt.legend(loc=location,ncol=ncols)
 leg.draw_frame(False)
 for x in leg.get_texts():
  x.set_fontsize(fsize)

def const_volume():
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "1"
    Angles = np.array([1.0,0.5,0.37,0.25,0.17,0.125,0.062])


    n_mle = 80
    n_mv = 80
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------

    Means_sum = np.array([])
    Std_sum = np.array([])
    
    Means_sum_weight = np.array([])
    Std_sum_weight = np.array([])

    Means_mle = np.array([])
    Std_mle = np.array([])
    Completeness_mle = np.array([])

    Means_mv = np.array([])
    Std_mv = np.array([])
    Completeness_mv = np.array([])

    #-------------------------------------- MAIN PROGRAM ------------------------------------

    for ang in Angles:
      Bulks_sum = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/Sum/Bulk_flows_sum_" + str(ang) + ".txt")
      Means_sum = np.append(Means_sum,np.mean(Bulks_sum))
      Std_sum = np.append(Std_sum,np.std(Bulks_sum))
      
      Bulks_sum_weight = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/SumWeight/Bulk_flows_sum_" + str(ang) + ".txt")
      Means_sum_weight = np.append(Means_sum_weight,np.mean(Bulks_sum_weight))
      Std_sum_weight = np.append(Std_sum_weight,np.std(Bulks_sum_weight))
  
      for i in np.arange(n_mle):
        Vectors_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Bulks_mle = np.sqrt( Vectors_mle[:,0]**2 + Vectors_mle[:,1]**2 + Vectors_mle[:,2]**2 )
        Completeness_mle = np.append(Completeness_mle,Bulks_mle)
      Means_mle = np.append(Means_mle,np.mean(Completeness_mle))
      Std_mle = np.append(Std_mle,np.std(Completeness_mle))
      Completeness_mle = np.array([])
  
      for i in np.arange(n_mv):
        Vectors_mv = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MV/MV_Bulk_flows_" + str(ang) + "_n300_" + str(i) +  ".txt")
        Bulks_mv = np.sqrt( Vectors_mv[:,0]**2 + Vectors_mv[:,1]**2 + Vectors_mv[:,2]**2 )
        Completeness_mv = np.append(Completeness_mv,Bulks_mv)
      Means_mv = np.append(Means_mv,np.mean(Completeness_mv))
      Std_mv = np.append(Std_mv,np.std(Completeness_mv))
      Completeness_mv = np.array([])

    #---------------------------------------- PLOTTING -------------------------------------

    plt.figure()
    #plt.title("Cosmic variance for constant-volume geometries")
    plt.xlim((0,360))
    plt.ylim((0,450))
    plt.xlabel(r"Angle - [$^{\circ}$]",size='xx-large')
    plt.ylabel(r"Bulk Flow Magnitude - [km$\,$s$^{-1}$]",size='xx-large')
    
    Angles = Angles * 360.0

    plt.fill_between(Angles,Means_mv+Std_mv,Means_mv-Std_mv,color='orange',alpha=0.6)
    plt.fill_between(Angles,Means_mle+Std_mle,Means_mle-Std_mle,color='blue',alpha=0.4)
    plt.fill_between(Angles,Means_sum+Std_sum,Means_sum-Std_sum,color='grey',alpha=0.5)
    #plt.fill_between(Angles,Means_sum_weight+Std_sum_weight,Means_sum_weight-Std_sum_weight,color='lightgrey',alpha=0.5)


    plt.plot(Angles,Means_mv+Std_mv,ls='-',marker='',lw=3,color='orange',label='MV',alpha=0.5)
    plt.plot(Angles,Means_mv,ls='--',marker='',lw=3,color='orange',alpha=0.2)
    plt.plot(Angles,Means_mv-Std_mv,ls='-',marker='',lw=3,color='orange',alpha=0.5)

    plt.plot(Angles,Means_mle+Std_mle,ls='-',marker='',lw=3,color='darkblue',label='MLE',alpha=0.5)
    plt.plot(Angles,Means_mle,ls='--',marker='',lw=3,color='darkblue',alpha=0.2)
    plt.plot(Angles,Means_mle-Std_mle,ls='-',marker='',lw=3,color='darkblue',alpha=0.5)

    plt.plot(Angles,Means_sum+Std_sum,ls='-',marker='',lw=3,color='k',label='Velocity Sum',alpha=0.5)
    plt.plot(Angles,Means_sum,ls='--',marker='',lw=3,color='k',alpha=0.2)
    plt.plot(Angles,Means_sum-Std_sum,ls='-',marker='',lw=3,color='k',alpha=0.5)
    
    #plt.plot(Angles,Means_sum_weight+Std_sum_weight,ls='-',marker='',lw=3,color='grey',label='Velocity Sum Weighted',alpha=0.5)
    #plt.plot(Angles,Means_sum_weight,ls='--',marker='',lw=3,color='grey',alpha=0.2)
    #plt.plot(Angles,Means_sum_weight-Std_sum_weight,ls='-',marker='',lw=3,color='grey',alpha=0.5)

    print Means_sum
    print Std_sum
    legend_plot(fsize=22)
    plt.subplots_adjust(bottom=0.12)
    plt.show()

def const_volume_new():
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "1"
    Angles = np.array([1.0,0.75,0.5,0.37,0.25,0.17,0.125])


    n_mle = 75
    n_mv = 75
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------

    Means_sum = np.array([])
    Std_sum = np.array([])

    Means_mle = np.array([])
    Std_mle = np.array([])
    Bulks_mle = np.array([])

    Means_mv = np.array([])
    Std_mv = np.array([])
    Bulks_mv = np.array([])

    #-------------------------------------- MAIN PROGRAM ------------------------------------

    for ang in Angles:
      Bulks_sum = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/Sum/Bulk_flows_sum_" + str(ang) + ".txt")
      Means_sum = np.append(Means_sum,np.mean(Bulks_sum))
      Std_sum = np.append(Std_sum,np.std(Bulks_sum))
  
      for i in np.arange(n_mle):
        Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Bulks_mle_x = Data_mle[:,0]
        Bulks_mle_y = Data_mle[:,1]
        Bulks_mle_z = Data_mle[:,2]
        Bulks_mle = np.append(Bulks_mle,np.sqrt( np.mean(Bulks_mle_x)**2 + np.mean(Bulks_mle_y)**2 + np.mean(Bulks_mle_z)**2) )
      Means_mle = np.append(Means_mle,np.mean(Bulks_mle))  
      Std_mle = np.append(Std_mle,np.std(Bulks_mle))
      Bulks_mle = np.array([])
  
      for i in np.arange(n_mv):
        Data_mv = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MV/MV_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Bulks_mv_x = Data_mv[:,0]
        Bulks_mv_y = Data_mv[:,1]
        Bulks_mv_z = Data_mv[:,2]
        Bulks_mv = np.append(Bulks_mv,np.sqrt( np.mean(Bulks_mv_x)**2 + np.mean(Bulks_mv_y)**2 + np.mean(Bulks_mv_z)**2) )
      Means_mv = np.append(Means_mv,np.mean(Bulks_mv))  
      Std_mv = np.append(Std_mv,np.std(Bulks_mv))
      Bulks_mv = np.array([])

    #---------------------------------------- PLOTTING -----------------------------------    
    
    Angles = Angles * 360.0
    
    plt.figure()
    plt.title("Cosmic variance for constant-volume geometries")
    plt.xlim((0,np.max(Angles)))
    plt.ylim((0,200))
    plt.xlabel(r"Angle",size='xx-large')
    plt.ylabel(r"Bulk Flow Magnitude - km$\cdot$s$^{-1}$",size='xx-large')

    plt.fill_between(Angles,Means_mv+Std_mv,Means_mv-Std_mv,color='orange',alpha=0.6)
    plt.fill_between(Angles,Means_mle+Std_mle,Means_mle-Std_mle,color='blue',alpha=0.4)
    plt.fill_between(Angles,Means_sum+Std_sum,Means_sum-Std_sum,color='grey',alpha=0.5)


    plt.plot(Angles,Means_mv+Std_mv,ls='-',marker='',lw=3,color='orange',label='MV',alpha=0.5)
    plt.plot(Angles,Means_mv,ls='--',marker='',lw=3,color='orange',alpha=0.2)
    plt.plot(Angles,Means_mv-Std_mv,ls='-',marker='',lw=3,color='orange',alpha=0.5)

    plt.plot(Angles,Means_mle+Std_mle,ls='-',marker='',lw=3,color='darkblue',label='MLE',alpha=0.5)
    plt.plot(Angles,Means_mle,ls='--',marker='',lw=3,color='darkblue',alpha=0.2)
    plt.plot(Angles,Means_mle-Std_mle,ls='-',marker='',lw=3,color='darkblue',alpha=0.5)

    plt.plot(Angles,Means_sum+Std_sum,ls='-',marker='',lw=3,color='k',label='Velocity Sum',alpha=0.5)
    plt.plot(Angles,Means_sum,ls='--',marker='',lw=3,color='k',alpha=0.2)
    plt.plot(Angles,Means_sum-Std_sum,ls='-',marker='',lw=3,color='k',alpha=0.5)

    print Means_sum
    print Std_sum
    legend_plot()
    plt.subplots_adjust(bottom=0.12)
    plt.show()

def shells():
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "4"
    R_outer = np.array([330,450,520,600,750,950])


    n_mle = 60
    n_mv = 60
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------

    Means_sum = np.array([])
    Std_sum = np.array([])

    Means_mle = np.array([])
    Std_mle = np.array([])
    Completeness_mle = np.array([])

    Means_mv = np.array([])
    Std_mv = np.array([])
    Completeness_mv = np.array([])

    #-------------------------------------- MAIN PROGRAM ------------------------------------

    for ang in R_outer:
      Bulks_sum = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/Sum/Bulk_flows_sum_" + str(ang) + ".txt")
      Means_sum = np.append(Means_sum,np.mean(Bulks_sum))
      Std_sum = np.append(Std_sum,np.std(Bulks_sum))
  
      for i in np.arange(n_mle):
        Bulks_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Completeness_mle = np.append(Completeness_mle,Bulks_mle)
      Means_mle = np.append(Means_mle,np.mean(Completeness_mle))
      Std_mle = np.append(Std_mle,np.std(Completeness_mle))
      Completeness_mle = np.array([])
  
      for i in np.arange(n_mv):
        Bulks_mv = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MV/MV_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Completeness_mv = np.append(Completeness_mv,Bulks_mv)
      Means_mv = np.append(Means_mv,np.mean(Completeness_mv))
      Std_mv = np.append(Std_mv,np.std(Completeness_mv))
      Completeness_mv = np.array([])

    #---------------------------------------- PLOTTING -------------------------------------

    plt.figure()
    #plt.title("Cosmic variance for constant-volume geometries")
    plt.xlim((330,950))
    plt.ylim((0,200))
    plt.xlabel(r"Outer Radius",size='xx-large')
    plt.ylabel(r"Bulk Flow Magnitude",size='xx-large')

    plt.fill_between(R_outer,Means_mv+Std_mv,Means_mv-Std_mv,color='orange',alpha=0.6)
    plt.fill_between(R_outer,Means_mle+Std_mle,Means_mle-Std_mle,color='blue',alpha=0.4)
    plt.fill_between(R_outer,Means_sum+Std_sum,Means_sum-Std_sum,color='grey',alpha=0.5)


    plt.plot(R_outer,Means_mv+Std_mv,ls='-',marker='',lw=3,color='orange',label='MV',alpha=0.5)
    plt.plot(R_outer,Means_mv,ls='--',marker='',lw=3,color='orange',alpha=0.2)
    plt.plot(R_outer,Means_mv-Std_mv,ls='-',marker='',lw=3,color='orange',alpha=0.5)

    plt.plot(R_outer,Means_mle+Std_mle,ls='-',marker='',lw=3,color='darkblue',label='MLE',alpha=0.5)
    plt.plot(R_outer,Means_mle,ls='--',marker='',lw=3,color='darkblue',alpha=0.2)
    plt.plot(R_outer,Means_mle-Std_mle,ls='-',marker='',lw=3,color='darkblue',alpha=0.5)

    plt.plot(R_outer,Means_sum+Std_sum,ls='-',marker='',lw=3,color='k',label='Velocity Sum',alpha=0.5)
    plt.plot(R_outer,Means_sum,ls='--',marker='',lw=3,color='k',alpha=0.2)
    plt.plot(R_outer,Means_sum-Std_sum,ls='-',marker='',lw=3,color='k',alpha=0.5)

    print Means_sum
    print Std_sum
    legend_plot()
    plt.show()

def shells_new():
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "4"
    R_outer = np.array([330,450,520,600,750,950])


    n_mle = 75
    n_mv = 0
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------

    Means_sum = np.array([])
    Std_sum = np.array([])

    Means_mle = np.array([])
    Std_mle = np.array([])
    Bulks_mle = np.array([])

    Means_mv = np.array([])
    Std_mv = np.array([])
    Completeness_mv = np.array([])

    #-------------------------------------- MAIN PROGRAM ------------------------------------

    for ang in R_outer:
      Bulks_sum = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/Sum/Bulk_flows_sum_" + str(ang) + ".txt")
      Means_sum = np.append(Means_sum,np.mean(Bulks_sum))
      Std_sum = np.append(Std_sum,np.std(Bulks_sum))
  
      for i in np.arange(n_mle):
        Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Bulks_mle_x = Data_mle[:,0]
        Bulks_mle_y = Data_mle[:,1]
        Bulks_mle_z = Data_mle[:,2]
        Bulks_mle = np.append(Bulks_mle,np.sqrt( np.mean(Bulks_mle_x)**2 + np.mean(Bulks_mle_y)**2 + np.mean(Bulks_mle_z)**2) )
      Means_mle = np.append(Means_mle,np.mean(Bulks_mle))  
      Std_mle = np.append(Std_mle,np.std(Bulks_mle))
      Bulks_mle = np.array([])
      
      '''    
      for i in np.arange(n_mv):
        Data_mv = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Bulks_mv_x = Data_mv[:,0]
        Bulks_mv_y = Data_mv[:,1]
        Bulks_mv_z = Data_mv[:,2]
        Bulks_mv = np.append(Bulks_mv,np.sqrt( np.mean(Bulks_mv_x)**2 + np.mean(Bulks_mv_y)**2 + np.mean(Bulks_mv_z)**2) )
      Means_mv = np.append(Means_mv,np.mean(Bulks_mv))  
      Std_mv = np.append(Std_mv,np.std(Bulks_mv))
      Bulks_mv = np.array([])
      '''

    #---------------------------------------- PLOTTING -------------------------------------

    plt.figure()
    plt.title("Cosmic variance for shell geometries")
    plt.xlim((330,950))
    plt.ylim((0,200))
    plt.xlabel(r"Outer Radius",size='xx-large')
    plt.ylabel(r"Bulk Flow Magnitude",size='xx-large')

    #plt.fill_between(R_outer,Means_mv+Std_mv,Means_mv-Std_mv,color='orange',alpha=0.6)
    plt.fill_between(R_outer,Means_mle+Std_mle,Means_mle-Std_mle,color='blue',alpha=0.4)
    plt.fill_between(R_outer,Means_sum+Std_sum,Means_sum-Std_sum,color='grey',alpha=0.5)


    #plt.plot(R_outer,Means_mv+Std_mv,ls='-',marker='',lw=3,color='orange',label='MV',alpha=0.5)
    #plt.plot(R_outer,Means_mv,ls='--',marker='',lw=3,color='orange',alpha=0.2)
    #plt.plot(R_outer,Means_mv-Std_mv,ls='-',marker='',lw=3,color='orange',alpha=0.5)

    plt.plot(R_outer,Means_mle+Std_mle,ls='-',marker='',lw=3,color='darkblue',label='MLE',alpha=0.5)
    plt.plot(R_outer,Means_mle,ls='--',marker='',lw=3,color='darkblue',alpha=0.2)
    plt.plot(R_outer,Means_mle-Std_mle,ls='-',marker='',lw=3,color='darkblue',alpha=0.5)

    plt.plot(R_outer,Means_sum+Std_sum,ls='-',marker='',lw=3,color='k',label='Velocity Sum',alpha=0.5)
    plt.plot(R_outer,Means_sum,ls='--',marker='',lw=3,color='k',alpha=0.2)
    plt.plot(R_outer,Means_sum-Std_sum,ls='-',marker='',lw=3,color='k',alpha=0.5)

    print Means_sum
    print Std_sum
    legend_plot()
    plt.show()


def spheres():
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "6"
    R_outer = np.array([330,450,520,600,750,950])


    n_mle = 60
    n_mv = 60
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------

    Means_sum = np.array([])
    Std_sum = np.array([])

    Means_mle = np.array([])
    Std_mle = np.array([])
    Completeness_mle = np.array([])

    Means_mv = np.array([])
    Std_mv = np.array([])
    Completeness_mv = np.array([])

    #-------------------------------------- MAIN PROGRAM ------------------------------------

    for ang in R_outer:
      Bulks_sum = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/Sum/Bulk_flows_sum_" + str(ang) + ".txt")
      Means_sum = np.append(Means_sum,np.mean(Bulks_sum))
      Std_sum = np.append(Std_sum,np.std(Bulks_sum))
  
      for i in np.arange(n_mle):
        Bulks_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Completeness_mle = np.append(Completeness_mle,Bulks_mle)
      Means_mle = np.append(Means_mle,np.mean(Completeness_mle))
      Std_mle = np.append(Std_mle,np.std(Completeness_mle))
      Completeness_mle = np.array([])
  
      for i in np.arange(n_mv):
        Bulks_mv = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MV/MV_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Completeness_mv = np.append(Completeness_mv,Bulks_mv)
      Means_mv = np.append(Means_mv,np.mean(Completeness_mv))
      Std_mv = np.append(Std_mv,np.std(Completeness_mv))
      Completeness_mv = np.array([])

    #---------------------------------------- PLOTTING -------------------------------------

    plt.figure()
    #plt.title("Cosmic variance for constant-volume geometries")
    plt.xlim((330,950))
    plt.ylim((0,200))
    plt.xlabel(r"Outer Radius",size='xx-large')
    plt.ylabel(r"Bulk Flow Magnitude",size='xx-large')

    plt.fill_between(R_outer,Means_mv+Std_mv,Means_mv-Std_mv,color='orange',alpha=0.6)
    plt.fill_between(R_outer,Means_mle+Std_mle,Means_mle-Std_mle,color='blue',alpha=0.4)
    plt.fill_between(R_outer,Means_sum+Std_sum,Means_sum-Std_sum,color='grey',alpha=0.5)


    plt.plot(R_outer,Means_mv+Std_mv,ls='-',marker='',lw=3,color='orange',label='MV',alpha=0.5)
    plt.plot(R_outer,Means_mv,ls='--',marker='',lw=3,color='orange',alpha=0.2)
    plt.plot(R_outer,Means_mv-Std_mv,ls='-',marker='',lw=3,color='orange',alpha=0.5)

    plt.plot(R_outer,Means_mle+Std_mle,ls='-',marker='',lw=3,color='darkblue',label='MLE',alpha=0.5)
    plt.plot(R_outer,Means_mle,ls='--',marker='',lw=3,color='darkblue',alpha=0.2)
    plt.plot(R_outer,Means_mle-Std_mle,ls='-',marker='',lw=3,color='darkblue',alpha=0.5)

    plt.plot(R_outer,Means_sum+Std_sum,ls='-',marker='',lw=3,color='k',label='Velocity Sum',alpha=0.5)
    plt.plot(R_outer,Means_sum,ls='--',marker='',lw=3,color='k',alpha=0.2)
    plt.plot(R_outer,Means_sum-Std_sum,ls='-',marker='',lw=3,color='k',alpha=0.5)

    print Means_sum
    print Std_sum
    legend_plot()
    plt.show()

def spheres_new():
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "6"
    R_outer = np.array([330,450,520,600,750,950])


    n_mle = 75
    n_mv = 0
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------

    Means_sum = np.array([])
    Std_sum = np.array([])

    Means_mle = np.array([])
    Std_mle = np.array([])
    Bulks_mle = np.array([])

    Means_mv = np.array([])
    Std_mv = np.array([])
    Completeness_mv = np.array([])

    #-------------------------------------- MAIN PROGRAM ------------------------------------

    for ang in R_outer:
      Bulks_sum = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/Sum/Bulk_flows_sum_" + str(ang) + ".txt")
      Means_sum = np.append(Means_sum,np.mean(Bulks_sum))
      Std_sum = np.append(Std_sum,np.std(Bulks_sum))
  
      for i in np.arange(n_mle):
        Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Bulks_mle_x = Data_mle[:,0]
        Bulks_mle_y = Data_mle[:,1]
        Bulks_mle_z = Data_mle[:,2]
        Bulks_mle = np.append(Bulks_mle,np.sqrt( np.mean(Bulks_mle_x)**2 + np.mean(Bulks_mle_y)**2 + np.mean(Bulks_mle_z)**2) )
      Means_mle = np.append(Means_mle,np.mean(Bulks_mle))  
      Std_mle = np.append(Std_mle,np.std(Bulks_mle))
      Bulks_mle = np.array([])
      
      '''    
      for i in np.arange(n_mv):
        Data_mv = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Bulks_mv_x = Data_mv[:,0]
        Bulks_mv_y = Data_mv[:,1]
        Bulks_mv_z = Data_mv[:,2]
        Bulks_mv = np.append(Bulks_mv,np.sqrt( np.mean(Bulks_mv_x)**2 + np.mean(Bulks_mv_y)**2 + np.mean(Bulks_mv_z)**2) )
      Means_mv = np.append(Means_mv,np.mean(Bulks_mv))  
      Std_mv = np.append(Std_mv,np.std(Bulks_mv))
      Bulks_mv = np.array([])
      '''

    #---------------------------------------- PLOTTING -------------------------------------

    plt.figure()
    plt.title("Cosmic variance for sphere geometries")
    plt.xlim((330,950))
    plt.ylim((0,200))
    plt.xlabel(r"Outer Radius",size='xx-large')
    plt.ylabel(r"Bulk Flow Magnitude",size='xx-large')

    #plt.fill_between(R_outer,Means_mv+Std_mv,Means_mv-Std_mv,color='orange',alpha=0.6)
    plt.fill_between(R_outer,Means_mle+Std_mle,Means_mle-Std_mle,color='blue',alpha=0.4)
    plt.fill_between(R_outer,Means_sum+Std_sum,Means_sum-Std_sum,color='grey',alpha=0.5)


    #plt.plot(R_outer,Means_mv+Std_mv,ls='-',marker='',lw=3,color='orange',label='MV',alpha=0.5)
    #plt.plot(R_outer,Means_mv,ls='--',marker='',lw=3,color='orange',alpha=0.2)
    #plt.plot(R_outer,Means_mv-Std_mv,ls='-',marker='',lw=3,color='orange',alpha=0.5)

    plt.plot(R_outer,Means_mle+Std_mle,ls='-',marker='',lw=3,color='darkblue',label='MLE',alpha=0.5)
    plt.plot(R_outer,Means_mle,ls='--',marker='',lw=3,color='darkblue',alpha=0.2)
    plt.plot(R_outer,Means_mle-Std_mle,ls='-',marker='',lw=3,color='darkblue',alpha=0.5)

    plt.plot(R_outer,Means_sum+Std_sum,ls='-',marker='',lw=3,color='k',label='Velocity Sum',alpha=0.5)
    plt.plot(R_outer,Means_sum,ls='--',marker='',lw=3,color='k',alpha=0.2)
    plt.plot(R_outer,Means_sum-Std_sum,ls='-',marker='',lw=3,color='k',alpha=0.5)

    print Means_sum
    print Std_sum
    legend_plot()
    plt.show()

def method_compar():

    root_dir = "/Users/perandersen/Data/"
    sub_dir = "1"
    
    Test_mv = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MV/MV_Bulk_flows_1.0_0.txt")
    Test_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_1.0_0.txt")

    n_mv = 60
    n_mle = 60
    i_offset = 0 #!!!Only for sampling variance. If set to other than 0 then n_mv and m_mle should be set to 1!!!

    Means_mv = []
    Stds_mv = []
    Means_mle = []
    Stds_mle = []

    Bulks_mv = np.zeros((n_mv,len(Test_mv)))
    for i in np.arange(n_mv):
     Data_mv = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MV/MV_Bulk_flows_1.0_" + str(i + i_offset) + ".txt")
     Bulks_mv[i,:] = Data_mv
     Means_mv.append(np.mean(Data_mv))
     Stds_mv.append(np.std(Data_mv))
     #plt.figure()
     #plt.hist(Bulks[i,:],10)
     #plt.show()
    Completeness_mv = Bulks_mv.reshape((np.shape(Bulks_mv)[0]*np.shape(Bulks_mv)[1],1))

    Bulks_mle = np.zeros((n_mle,len(Test_mle)))
    for i in np.arange(n_mle):
     Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_1.0_" + str(i + i_offset) + ".txt")
     Bulks_mle[i,:] = Data_mle
     Means_mle.append(np.mean(Data_mle))
     Stds_mle.append(np.std(Data_mle))
     #plt.figure()
     #plt.hist(Bulks[i,:],10)
     #plt.show()
    Completeness_mle = Bulks_mle.reshape((np.shape(Bulks_mle)[0]*np.shape(Bulks_mle)[1],1))


    plt.figure()
    plt.title("HR2 Cosmic Variance")
    plt.xlabel(r"Bulk Flow Velocity [km/s]",size='x-large')
    plt.ylabel(r"Normalised Probability",size='xx-large')
    plt.xlim((0,400))
    plt.hist(Completeness_mle,range=(0,400),bins=20,normed=True,edgecolor='none',color='dimgrey',label="MLE")
    plt.hist(Completeness_mv,range=(0,400),bins=20,normed=True,edgecolor='none',color='darkgrey',label="MV")
    plt.hist(Completeness_mle,range=(0,400),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot()

    print "Cosmic Variance"
    print "Mean MLE/MV: ", np.mean(Completeness_mle), "/", np.mean(Completeness_mv)
    print "Std  MLE/MV: ", np.std(Completeness_mle), "/", np.std(Completeness_mv)

    plt.show()

def sampling():

    root_dir = "/Users/perandersen/Data/"
    sub_dir = "5"
    
    N_per_mle = [500,100,50]
    Colors = ['lightgrey','darkgrey','dimgrey']
    plt.figure()
    for i in np.arange(len(N_per_mle)):
  
      Means_mle = []
      Stds_mle = []
      
      Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.5_" + str(N_per_mle[i]) + ".txt")
      Bulks_mle = np.sqrt(Data_mle[:,0]**2 + Data_mle[:,1]**2 + Data_mle[:,2]**2)
      mean_mle = np.mean(Bulks_mle)
      std_mle = np.std(Bulks_mle)
      
      
      print "HR2 MLE Sampling Variance"
      print "n: ", N_per_mle[i]
      print "MLE: ", np.mean(Bulks_mle), "+/-", np.std(Bulks_mle)
  
      
      plt.title("HR2 Sampling Variance")
      plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
      plt.ylabel(r"Normalised Probability",size='xx-large')
      plt.xlim((0,500))
      plt.hist(Bulks_mle,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors[i],label="MLE " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    plt.show()

def sampling_new():

    root_dir = "/Users/perandersen/Data/"
    sub_dir = "5"
    
    n_per_mle = 100
    
    Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_1_" + str(n_per_mle) + ".txt")
    #Data_mle = Data_mle[:1000]
    Vx = Data_mle[:,0]
    Vy = Data_mle[:,1]
    Vz = Data_mle[:,2]
    
    
    print "HR2 Sampling Variance"
    print "n: ", n_per_mle
    print "Vx: ", np.mean(Vx), " +/- ", np.std(Vx)
    print "Vy: ", np.mean(Vy), " +/- ", np.std(Vy)
    print "Vz: ", np.mean(Vz), " +/- ", np.std(Vz)
    print "V: ", np.sqrt(np.mean(Vx)**2 + np.mean(Vy)**2 + np.mean(Vz)**2)


def sampling025():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "7"
    
    N_per_mle = [500,100,50]
    Colors = ['lightgrey','darkgrey','dimgrey']
    
    plt.figure()
    print "HR2 MLE Sampling Variance"
    for i in np.arange(len(N_per_mle)):
  
      Means_mle = []
      Stds_mle = []
      
      Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_0_" + str(N_per_mle[i]) + ".txt")
      Bulks_mle = np.sqrt(Data_mle[:,0]**2 + Data_mle[:,1]**2 + Data_mle[:,2]**2)
      mean_mle = np.mean(Bulks_mle)
      std_mle = np.std(Bulks_mle)
      
      
      print "n: ", N_per_mle[i], "->",np.mean(Bulks_mle), "+/-", np.std(Bulks_mle)
  
      
      plt.title("HR2 Sampling Variance")
      plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
      plt.ylabel(r"Normalised Probability",size='xx-large')
      plt.xlim((0,500))
      plt.hist(Bulks_mle,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)
    
def completeness025():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "7"
    
    N_per_mle = [500,100,50]
    n_rot = 80
    n_bulk = 2000
    Bulks = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    Colors = ['lightgrey','darkgrey','dimgrey']
    
    for i in np.arange(n_rot):
        for j in np.arange(len(N_per_mle)):
            Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
            Bulks[i,j,:] = np.sqrt(Data[:,0]**2 + Data[:,1]**2 + Data[:,2]**2)
    
    plt.figure()
    print "HR2 MLE Completeness Variance"
    for i in np.arange(len(N_per_mle)):
        Completeness = Bulks[:,i,:]
        Completeness = Completeness.reshape((n_rot*n_bulk))
        
        print "n: ", N_per_mle[i], "->",np.mean(Completeness), "+/-", np.std(Completeness)
    
        
        plt.title("HR2 Completeness Variance")
        plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
        plt.ylabel(r"Normalised Probability",size='xx-large')
        plt.xlim((0,500))
        plt.hist(Completeness,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
        plt.hist(Completeness,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)

def cosmic025():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "8"
    
    #N_per_mle = [1000,500,100,50]
    N_per_mle = [2000,1000,500,100,50]
    n_rot = 80
    n_bulk = 2000
    Bulks = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    Colors = ['lightgrey','darkgrey','grey','dimgrey','#3F3F3F']
    
    for i in np.arange(n_rot):
        for j in np.arange(len(N_per_mle)):
            Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
            Bulks[i,j,:] = np.sqrt(Data[:,0]**2 + Data[:,1]**2 + Data[:,2]**2)
    
    
    #Calculating theoretical prediction from linear theory
    
    #Setting constants
    hubble_const = 100#67.0 #Set to 100 to have units in Mpc/h
    o_m = 0.3175
    f = o_m**0.55
    
    root_dir = "/Users/perandersen/Data/"
    #Reading power spectrum data !!!CHANGE THIS FOR YOUR FILESYSTEM!!!
    Power = np.genfromtxt(root_dir + "BulkFlow/DataCommon/pk_planck_z0_nonlinear_matterpower.dat")

    #Here the power spectrum is sliced into K and P_k arrays
    K = Power[:,0]
    P_k = Power[:,1]
    
    r_bulk = 320. #Radius to calculate theoretical prediction for. Using effective radius.
    
    #This is the fourier transform of the window function
    W = np.exp(- (K*r_bulk)**2.0 / 2.0)
    
    #This is the integral over the power spectrum from eq (3) in Per's bulk flow paper
    vel_disp_sq = np.trapz(P_k*W**2,x=K) * hubble_const**2 * f**2 / 19.739208802 #19.739 = 2*pi^2
    vel_disp = np.sqrt(vel_disp_sq)
    vel = vel_disp * np.sqrt(2.0/3.0)
    
    plt.figure()
    print "HR2 MLE Cosmic Variance"
    for i in np.arange(len(N_per_mle)):
        Completeness = Bulks[:,i,:]
        Completeness = Completeness.reshape((n_rot*n_bulk))
        
        print "n: ", N_per_mle[i], "->",np.mean(Completeness), "+/-", np.std(Completeness)
    
        
        plt.title("HR2 Cosmic Variance")
        plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
        plt.ylabel(r"Normalised Probability",size='xx-large')
        plt.xlim((0,500))
        plt.hist(Completeness,range=(0,500),bins=85,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
        plt.hist(Completeness,range=(0,500),bins=85,normed=True,histtype='stepfilled',fill=False)
    plt.axvline(vel,color='r',lw = 2.)
    plt.axvspan(vel - 0.356*vel_disp, vel + 0.419*vel_disp,alpha=0.4,color='r',label="LinTheo")
    plt.axvspan(vel - 0.619*vel_disp, vel + 0.891*vel_disp,alpha=0.3,color='r')
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)


def sampling025_vector():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "7"
    
    N_per_mle = [50,100,500]
    Colors = ['lightgrey','darkgrey','dimgrey']
    
    
    print "HR2 MLE Sampling Variance"

    for i in np.arange(len(N_per_mle)):
      
      Means_mle = []
      Stds_mle = []
      
      Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_0_" + str(N_per_mle[i]) + ".txt")
      
      Bulks_mle_x = Data_mle[:,0]
      mean_mle_x = np.mean(Bulks_mle_x)
      std_mle_x = np.std(Bulks_mle_x)
      
      Bulks_mle_y = Data_mle[:,1]
      mean_mle_y = np.mean(Bulks_mle_y)
      std_mle_y = np.std(Bulks_mle_y)
      
      Bulks_mle_z = Data_mle[:,2]
      mean_mle_z = np.mean(Bulks_mle_z)
      std_mle_z = np.std(Bulks_mle_z)
      
      
      print "n: ", N_per_mle[i]
      print np.mean(Bulks_mle_x), "+/-", np.std(Bulks_mle_x)
      print np.mean(Bulks_mle_y), "+/-", np.std(Bulks_mle_y)
      print np.mean(Bulks_mle_z), "+/-", np.std(Bulks_mle_z)
  
      plt.subplot(3,1,1)
      plt.title("HR2 Sampling Variance Vector")
      plt.hist(Bulks_mle_x,range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle_x,range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
      plt.xlabel("X")
      legend_plot(fsize=12)
      plt.subplot(3,1,2)
      plt.hist(Bulks_mle_y,range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle_y,range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
      plt.xlabel("Y")
      legend_plot(fsize=12)
      plt.subplot(3,1,3)
      plt.hist(Bulks_mle_z,range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle_z,range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
      plt.xlabel("Z")
      legend_plot(fsize=12)
    plt.subplots_adjust(bottom=0.10,hspace=0.35)
    plt.xlim((-300,300))
            
def completeness025_vector():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "7"
    
    N_per_mle = [50,100,500]
    n_rot = 75
    n_bulk = 2000
    Colors = ['lightgrey','darkgrey','dimgrey']
    Completeness_x = np.zeros(  (n_rot * n_bulk, len(N_per_mle))  )
    Completeness_y = np.zeros(  (n_rot * n_bulk, len(N_per_mle))  )
    Completeness_z = np.zeros(  (n_rot * n_bulk, len(N_per_mle))  )
    Completeness_bulk = np.zeros(  (n_rot, len(N_per_mle))  )
    
    for i in np.arange(len(N_per_mle)):
        for j in np.arange(n_rot):
            Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_" + str(j) + "_" + str(N_per_mle[i]) + ".txt")
            Completeness_x[j*n_bulk:(j+1)*n_bulk,i] = Data[:,0]
            Completeness_y[j*n_bulk:(j+1)*n_bulk,i] = Data[:,1]
            Completeness_z[j*n_bulk:(j+1)*n_bulk,i] = Data[:,2]
            vx_mean = np.mean(Data[:,0])
            vy_mean = np.mean(Data[:,1])
            vz_mean = np.mean(Data[:,2])
            Completeness_bulk[j,i] = np.sqrt(  vx_mean**2 + vy_mean**2 + vz_mean**2  )
            
    plt.figure()
    plt.title("HR Completeness Variance Vector")
    plt.xlabel("X")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_x[:,i],range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_x[:,i],range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    
    plt.figure()
    plt.title("HR Completeness Variance Vector")
    plt.xlabel("Y")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_y[:,i],range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_y[:,i],range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    
    plt.figure()
    plt.title("HR Completeness Variance Vector")
    plt.xlabel("Z")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_z[:,i],range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_z[:,i],range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    
    plt.figure()
    plt.title("HR Completeness Variance Vector")
    plt.xlabel("Mean Bulk Flow")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_bulk[:,i],range=(0,250),bins=10,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_bulk[:,i],range=(0,250),bins=10,normed=True,histtype='stepfilled',fill=False)    
    legend_plot(fsize=25)

def cosmic025_vector():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "8"
    
    N_per_mle = [50,100,500]
    n_rot = 75
    n_bulk = 2000
    Colors = ['lightgrey','darkgrey','dimgrey']
    Completeness_x = np.zeros(  (n_rot * n_bulk, len(N_per_mle))  )
    Completeness_y = np.zeros(  (n_rot * n_bulk, len(N_per_mle))  )
    Completeness_z = np.zeros(  (n_rot * n_bulk, len(N_per_mle))  )
    Completeness_bulk = np.zeros(  (n_rot, len(N_per_mle))  )
    
    for i in np.arange(len(N_per_mle)):
        for j in np.arange(n_rot):
            Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_" + str(j) + "_" + str(N_per_mle[i]) + ".txt")
            Completeness_x[j*n_bulk:(j+1)*n_bulk,i] = Data[:,0]
            Completeness_y[j*n_bulk:(j+1)*n_bulk,i] = Data[:,1]
            Completeness_z[j*n_bulk:(j+1)*n_bulk,i] = Data[:,2]
            vx_mean = np.mean(Data[:,0])
            vy_mean = np.mean(Data[:,1])
            vz_mean = np.mean(Data[:,2])
            Completeness_bulk[j,i] = np.sqrt(  vx_mean**2 + vy_mean**2 + vz_mean**2  )
    
    #Calculating theoretical prediction from linear theory
    
    #Setting constants
    hubble_const = 100#67.0 #Set to 100 to have units in Mpc/h
    o_m = 0.3175
    f = o_m**0.55
    
    root_dir = "/Users/perandersen/Data/"
    #Reading power spectrum data !!!CHANGE THIS FOR YOUR FILESYSTEM!!!
    Power = np.genfromtxt(root_dir + "BulkFlow/DataCommon/pk_planck_z0_nonlinear_matterpower.dat")

    #Here the power spectrum is sliced into K and P_k arrays
    K = Power[:,0]
    P_k = Power[:,1]
    
    r_bulk = 320. #Radius to calculate theoretical prediction for. Using effective radius.
    
    #This is the fourier transform of the window function
    W = np.exp(- (K*r_bulk)**2.0 / 2.0)
    
    #This is the integral over the power spectrum from eq (3) in Per's bulk flow paper
    vel_disp_sq = np.trapz(P_k*W**2,x=K) * hubble_const**2 * f**2 / 19.739208802 #19.739 = 2*pi^2
    vel_disp = np.sqrt(vel_disp_sq)
    vel = vel_disp * np.sqrt(2.0/3.0)
    
    plt.figure()
    plt.title("HR Cosmic Variance Vector")
    plt.xlabel("X")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_x[:,i],range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_x[:,i],range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    
    plt.figure()
    plt.title("HR Cosmic Variance Vector")
    plt.xlabel("Y")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_y[:,i],range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_y[:,i],range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    
    plt.figure()
    plt.title("HR Cosmic Variance Vector")
    plt.xlabel("Z")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_z[:,i],range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_z[:,i],range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    
    plt.figure()
    plt.title("HR Cosmic Variance Vector")
    plt.xlabel("Mean Bulk Flow")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_bulk[:,i],range=(0,300),bins=10,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_bulk[:,i],range=(0,300),bins=10,normed=True,histtype='stepfilled',fill=False)
    plt.axvline(vel,color='r',lw = 2.)
    plt.axvspan(vel - 0.356*vel_disp, vel + 0.419*vel_disp,alpha=0.4,color='r',label="LinTheo")
    plt.axvspan(vel - 0.619*vel_disp, vel + 0.891*vel_disp,alpha=0.3,color='r')
    legend_plot(fsize=25)
    

def sampling025_SDSS():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "9"
    
    #N_per_mle = [50,100,500,2000]
    N_per_mle = [500,100,50]
    Colors = 'lightgrey','darkgrey','grey','dimgrey','#3F3F3F'
    
    plt.figure()
    print "HR2 MLE Sampling Variance"
    for i in np.arange(len(N_per_mle)):
  
      Means_mle = []
      Stds_mle = []
      
      Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_0_" + str(N_per_mle[i]) + ".txt")
      Bulks_mle = np.sqrt(Data_mle[:,0]**2 + Data_mle[:,1]**2 + Data_mle[:,2]**2)
      mean_mle = np.mean(Bulks_mle)
      std_mle = np.std(Bulks_mle)
      
      
      print "n: ", N_per_mle[i], "->",np.mean(Bulks_mle), "+/-", np.std(Bulks_mle)
  
      
      plt.title("HR2 Sampling Variance - SDSSIII Mock")
      plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
      plt.ylabel(r"Normalised Probability",size='xx-large')
      plt.xlim((0,500))
      plt.hist(Bulks_mle,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)
def completeness025_SDSS():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "9"
    
    #N_per_mle = [50,100,500,2000]
    N_per_mle = [500,100,50]
    Colors = 'lightgrey','darkgrey','grey','dimgrey','#3F3F3F'
    
    n_rot = 80
    n_bulk = 2000
    Bulks = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    
    for i in np.arange(n_rot):
        for j in np.arange(len(N_per_mle)):
            Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
            Bulks[i,j,:] = np.sqrt(Data[:,0]**2 + Data[:,1]**2 + Data[:,2]**2)
    
    plt.figure()
    print "HR2 MLE Completeness Variance"
    for i in np.arange(len(N_per_mle)):
        Completeness = Bulks[:,i,:]
        Completeness = Completeness.reshape((n_rot*n_bulk))
        
        print "n: ", N_per_mle[i], "->",np.mean(Completeness), "+/-", np.std(Completeness)
    
        
        plt.title("HR2 Completeness Variance - SDSSIII Mock")
        plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
        plt.ylabel(r"Normalised Probability",size='xx-large')
        plt.xlim((0,500))
        plt.hist(Completeness,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
        plt.hist(Completeness,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)
    
def sampling025_vector_SDSS():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "9"
    
    N_per_mle = [50,100,500,1000,2000]
    Colors = 'lightgrey','darkgrey','grey','dimgrey','#3F3F3F'
    
    
    print "HR2 MLE Sampling Variance - SDSSIII Mock"

    for i in np.arange(len(N_per_mle)):
      
      Means_mle = []
      Stds_mle = []
      
      Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_0_" + str(N_per_mle[i]) + ".txt")
      
      Bulks_mle_x = Data_mle[:,0]
      mean_mle_x = np.mean(Bulks_mle_x)
      std_mle_x = np.std(Bulks_mle_x)
      
      Bulks_mle_y = Data_mle[:,1]
      mean_mle_y = np.mean(Bulks_mle_y)
      std_mle_y = np.std(Bulks_mle_y)
      
      Bulks_mle_z = Data_mle[:,2]
      mean_mle_z = np.mean(Bulks_mle_z)
      std_mle_z = np.std(Bulks_mle_z)
      
      
      print "n: ", N_per_mle[i]
      print np.mean(Bulks_mle_x), "+/-", np.std(Bulks_mle_x)
      print np.mean(Bulks_mle_y), "+/-", np.std(Bulks_mle_y)
      print np.mean(Bulks_mle_z), "+/-", np.std(Bulks_mle_z)
  
      plt.subplot(3,1,1)
      plt.title("HR2 Sampling Variance Vector")
      plt.hist(Bulks_mle_x,range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle_x,range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
      plt.xlabel("X")
      legend_plot(fsize=12)
      plt.subplot(3,1,2)
      plt.hist(Bulks_mle_y,range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle_y,range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
      plt.xlabel("Y")
      legend_plot(fsize=12)
      plt.subplot(3,1,3)
      plt.hist(Bulks_mle_z,range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle_z,range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
      plt.xlabel("Z")
      legend_plot(fsize=12)
    plt.subplots_adjust(bottom=0.10,hspace=0.35)
    plt.xlim((-300,300))        
def completeness025_vector_SDSS():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "9"
    
    
    n_rot = 75
    n_bulk = 2000
    
    N_per_mle = [50,100,500,1000,2000]
    Colors = 'lightgrey','darkgrey','grey','dimgrey','#3F3F3F'
    
    Completeness_x = np.zeros(  (n_rot * n_bulk, len(N_per_mle))  )
    Completeness_y = np.zeros(  (n_rot * n_bulk, len(N_per_mle))  )
    Completeness_z = np.zeros(  (n_rot * n_bulk, len(N_per_mle))  )
    Completeness_bulk = np.zeros(  (n_rot, len(N_per_mle))  )
    
    for i in np.arange(len(N_per_mle)):
        for j in np.arange(n_rot):
            Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_" + str(j) + "_" + str(N_per_mle[i]) + ".txt")
            Completeness_x[j*n_bulk:(j+1)*n_bulk,i] = Data[:,0]
            Completeness_y[j*n_bulk:(j+1)*n_bulk,i] = Data[:,1]
            Completeness_z[j*n_bulk:(j+1)*n_bulk,i] = Data[:,2]
            vx_mean = np.mean(Data[:,0])
            vy_mean = np.mean(Data[:,1])
            vz_mean = np.mean(Data[:,2])
            Completeness_bulk[j,i] = np.sqrt(  vx_mean**2 + vy_mean**2 + vz_mean**2  )
            
    plt.figure()
    plt.title("HR Completeness Variance Vector - SDSSIII Mock")
    plt.xlabel("X")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_x[:,i],range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_x[:,i],range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    
    plt.figure()
    plt.title("HR Completeness Variance Vector - SDSSIII Mock")
    plt.xlabel("Y")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_y[:,i],range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_y[:,i],range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    
    plt.figure()
    plt.title("HR Completeness Variance Vector - SDSSIII Mock")
    plt.xlabel("Z")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_z[:,i],range=(-300,300),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_z[:,i],range=(-300,300),bins=20,normed=True,histtype='stepfilled',fill=False)
    legend_plot(fsize=25)
    
    plt.figure()
    plt.title("HR Completeness Variance Vector - SDSSIII Mock")
    plt.xlabel("Mean Bulk Flow")
    for i in np.arange(len(N_per_mle)):
      plt.hist(Completeness_bulk[:,i],range=(0,250),bins=10,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Completeness_bulk[:,i],range=(0,250),bins=10,normed=True,histtype='stepfilled',fill=False)    
    legend_plot(fsize=25)
    
def sampling025_FoF():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "11"
    
    #N_per_mle = [50,100,500,2000]
    N_per_mle = [500,100,50]
    Colors = 'lightgrey','darkgrey','grey','dimgrey','#3F3F3F'
    
    plt.figure()
    print "HR2 MLE Sampling Variance"
    for i in np.arange(len(N_per_mle)):
  
      Means_mle = []
      Stds_mle = []
      
      Data_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_0_" + str(N_per_mle[i]) + ".txt")
      Bulks_mle = np.sqrt(Data_mle[:,0]**2 + Data_mle[:,1]**2 + Data_mle[:,2]**2)
      mean_mle = np.mean(Bulks_mle)
      std_mle = np.std(Bulks_mle)
      
      
      print "n: ", N_per_mle[i], "->",np.mean(Bulks_mle), "+/-", np.std(Bulks_mle)
  
      
      plt.title("HR2 Sampling Variance - FoF DM Halo")
      plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
      plt.ylabel(r"Normalised Probability",size='xx-large')
      plt.xlim((0,500))
      plt.hist(Bulks_mle,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)    
def completeness025_FoF():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "11"
    
    #N_per_mle = [50,100,500,2000]
    N_per_mle = [500,100,50]
    Colors = 'lightgrey','darkgrey','grey','dimgrey','#3F3F3F'
    
    n_rot = 80
    n_bulk = 2000
    Bulks = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    
    for i in np.arange(n_rot):
        for j in np.arange(len(N_per_mle)):
            Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.25_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
            Bulks[i,j,:] = np.sqrt(Data[:,0]**2 + Data[:,1]**2 + Data[:,2]**2)
    
    plt.figure()
    print "HR2 MLE Completeness Variance"
    for i in np.arange(len(N_per_mle)):
        Completeness = Bulks[:,i,:]
        Completeness = Completeness.reshape((n_rot*n_bulk))
        
        print "n: ", N_per_mle[i], "->",np.mean(Completeness), "+/-", np.std(Completeness)
    
        
        plt.title("HR2 Completeness Variance - FoF DM Halo")
        plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
        plt.ylabel(r"Normalised Probability",size='xx-large')
        plt.xlim((0,500))
        plt.hist(Completeness,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
        plt.hist(Completeness,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)

def samp025_FoF_mock():
    root_dir = "/Users/perandersen/Data/"
    sub_dir_mock = "9"
    sub_dir_fof = "11"
    
    #N_per_mle = [50,100,500,2000]
    N_per_mle = [500,100,50]
    Colors_fof = 'lightskyblue','darkblue','cornflowerblue','dimgrey','#3F3F3F'
    Colors_mock = 'rosybrown','firebrick','r','dimgrey','#3F3F3F'
    
    plt.figure()
    for i in np.arange(len(N_per_mle)):
  
      Means_mle_mock = []
      Stds_mle_mock = []
      
      Data_mle_mock = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir_mock + "/MLE/MLE_Bulk_flows_0.25_0_" + str(N_per_mle[i]) + ".txt")
      Bulks_mle_mock = np.sqrt(Data_mle_mock[:,0]**2 + Data_mle_mock[:,1]**2 + Data_mle_mock[:,2]**2)
      mean_mle_mock = np.mean(Bulks_mle_mock)
      std_mle_mock = np.std(Bulks_mle_mock)
  
      Means_mle_fof = []
      Stds_mle_fof = []
      
      Data_mle_fof = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir_fof + "/MLE/MLE_Bulk_flows_0.25_0_" + str(N_per_mle[i]) + ".txt")
      Bulks_mle_fof = np.sqrt(Data_mle_fof[:,0]**2 + Data_mle_fof[:,1]**2 + Data_mle_fof[:,2]**2)
      mean_mle_fof = np.mean(Bulks_mle_fof)
      std_mle_fof = np.std(Bulks_mle_fof)
      
      plt.title("HR2 Sampling Variance - SDSSIII Mock and FoF DM Halo")
      plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
      plt.ylabel(r"Normalised Probability",size='xx-large')
      plt.xlim((0,500))
      plt.hist(Bulks_mle_mock,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors_mock[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle_mock,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
      plt.hist(Bulks_mle_fof,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors_fof[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
      plt.hist(Bulks_mle_fof,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)

def comp025_FoF_mock():
    root_dir = "/Users/perandersen/Data/"
    sub_dir_mock = "9"
    sub_dir_fof = "11"
    
    #N_per_mle = [50,100,500,2000]
    N_per_mle = [500,100,50]
    Colors_fof = 'lightskyblue','darkblue','cornflowerblue','dimgrey','#3F3F3F'
    Colors_mock = 'rosybrown','firebrick','r','dimgrey','#3F3F3F'
    
    n_rot = 80
    n_bulk = 2000
    Bulks_mock = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    Bulks_fof = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    
    for i in np.arange(n_rot):
        for j in np.arange(len(N_per_mle)):
            Data_mock = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir_mock + "/MLE/MLE_Bulk_flows_0.25_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
            Bulks_mock[i,j,:] = np.sqrt(Data_mock[:,0]**2 + Data_mock[:,1]**2 + Data_mock[:,2]**2)
            Data_fof = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir_fof + "/MLE/MLE_Bulk_flows_0.25_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
            Bulks_fof[i,j,:] = np.sqrt(Data_fof[:,0]**2 + Data_fof[:,1]**2 + Data_fof[:,2]**2)
    
    plt.figure()
    for i in np.arange(len(N_per_mle)):
        Completeness_mock = Bulks_mock[:,i,:]
        Completeness_mock = Completeness_mock.reshape((n_rot*n_bulk))
        Completeness_fof = Bulks_fof[:,i,:]
        Completeness_fof = Completeness_fof.reshape((n_rot*n_bulk))
    
        
        plt.title("HR2 Completeness Variance - SDSSIII Mock and FoF DM Halo")
        plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
        plt.ylabel(r"Normalised Probability",size='xx-large')
        plt.xlim((0,500))
        plt.hist(Completeness_mock,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors_mock[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
        plt.hist(Completeness_mock,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
        plt.hist(Completeness_fof,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors_fof[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
        plt.hist(Completeness_fof,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)

     
#shells()
#shells_new()
#spheres()
#spheres_new()

const_volume()

#const_volume_new()
#method_compar()
#sampling()
#sampling_new()

#sampling025()
#completeness025()
#cosmic025()

#sampling025_vector()
#completeness025_vector()
#cosmic025_vector()

#sampling025_SDSS()
#completeness025_SDSS()
#sampling025_vector_SDSS()
#completeness025_vector_SDSS()

#sampling025_FoF()
#completeness025_FoF()

#samp025_FoF_mock()
#comp025_FoF_mock()
plt.show()