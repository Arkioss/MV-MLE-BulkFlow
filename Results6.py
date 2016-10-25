from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from scipy.stats import chisquare
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from scipy.integrate import quad

#---------------------------------- DEFINING FUNCTIONS ----------------------------------
def lin_interp(X,Y,x):
	#Sanity check - lengths should be identical for X and Y
 	if (len(X) != len(Y)): 
  		print "Mismatched input lengts!", exit()
 
 	#Sanity check - we need to be interpolating, not exterpolating
 	if ( (x > np.max(X)) or (x < np.min(X)) ):
  		print np.min(X), x, np.max(X), "Interpolation impossible!"
  		raise
 
 	#Index to use for interpolation
 	i = 0
 
 	#Finding correct index for interpolation
 	while (X[i] < x):
  		i += 1
 
 	#Doing linear interpolation
 	if i == 0:
  		return Y[i] + (Y[i+1]-Y[i]) * (x - X[i]) / (X[i+1] - X[i])
 	else:
  		return Y[i-1] + (Y[i]-Y[i-1]) * (x - X[i-1]) / (X[i] - X[i-1])

def legend_plot(location=1,fsize=18,ncols=1):
	leg = plt.legend(loc=location,ncol=ncols,numpoints=1)
	leg.draw_frame(False)
	for x in leg.get_texts():
		x.set_fontsize(fsize)

def maxwell(v, sig_v):
    return np.sqrt(2. / np.pi) * (3. / (sig_v**2) )**(1.5) * v * v * np.exp(-3*v*v / (2.*sig_v*sig_v))

def maxwell_extra(x, stretch, factor, sig_v):
    v = x * ( 1. + stretch)
    return ( np.sqrt(2. / np.pi) * (3. / (sig_v**2) )**(1.5) * v * v * np.exp(-3*v*v / (2.*sig_v*sig_v)) ) * factor

def residuals(p,x,y):
    
    penalty1 = abs(1. - p[1]) * 0.001
    penalty2 = abs(0. - p[0]) * 0.0001
    return abs(y - maxwell_extra(x, p[0], p[1], p[2])) + penalty2 + penalty1

def max_likehood_find(Sample,nbins):
    N, Bins = np.histogram(Sample,range=(0,np.max(Sample)),bins=nbins,density=False)
    Bins_mid_points = (0.5*(Bins + np.roll(Bins, 1)))[1:]
    N = N / np.max(N)
    N_norm, Bins = np.histogram(Sample,range=(0,np.max(Sample)),bins=nbins,density=True)
    Bins_mid_plot_points = np.linspace(np.min(Bins_mid_points), np.max(Bins_mid_points), 5000)
      
    #The curve fit method is executed here - not used because it was hard to impose
    #a limit on the guesses, and without them the method gave strange values.
    '''
    start_guess = (0., 1., 300)
    Popt1, Pcov1 = curve_fit(maxwell_extra, xdata=Bins_mid_points, ydata=N_norm,p0=start_guess)
    Probability_curve_fit = maxwell_extra(Bins_mid_plot_points, *Popt1)
    max_like_curve_fit = Bins_mid_plot_points[np.argmax(Probability_curve_fit)]
    max_prop_curve_fit = np.max(Probability_curve_fit)

    prop_sum_curve_fit = 0.
    upper_limit_curve_fit = max_like_curve_fit
    while (prop_sum_curve_fit < 0.34135):
      prop_sum_curve_fit = quad(maxwell_extra, max_like_curve_fit, upper_limit_curve_fit, args=(Popt1[0], Popt1[1], Popt1[2]))[0]
      upper_limit_curve_fit += 0.1

    upper_prop_curve_fit = maxwell_extra(upper_limit_curve_fit, *Popt1)

    prop_sum_curve_fit = 0.
    lower_limit_curve_fit = max_like_curve_fit
    while (prop_sum_curve_fit < 0.34135):
        prop_sum_curve_fit = quad(maxwell_extra, lower_limit_curve_fit, max_like_curve_fit, args=(Popt1[0], Popt1[1], Popt1[2]))[0]
        lower_limit_curve_fit -= 0.1

    lower_prop_curve_fit = maxwell_extra(lower_limit_curve_fit, *Popt1)
    '''

    #The least squares method is executed here
    Popt2, Pcov2 = leastsq(func=residuals, x0=(0.,1.,300.), args=(Bins_mid_points, N_norm))
    Probability_least_sq  = maxwell_extra(Bins_mid_plot_points, *Popt2)
    max_like_least_sq =  Bins_mid_plot_points[np.argmax(Probability_least_sq)]
    max_prop_least_sq = np.max(Probability_least_sq)
    prop_sum_least_sq = 0.
    upper_limit_least_sq = max_like_least_sq
    while (prop_sum_least_sq < 0.34135):
      prop_sum_least_sq = quad(maxwell_extra, max_like_least_sq, upper_limit_least_sq, args=(Popt2[0], Popt2[1], Popt2[2]))[0]
      upper_limit_least_sq += 0.1

    upper_prop_least_sq = maxwell_extra(upper_limit_least_sq, *Popt2)

    prop_sum_least_sq = 0.
    lower_limit_least_sq = max_like_least_sq
    while (prop_sum_least_sq < 0.34135):
      prop_sum_least_sq = quad(maxwell_extra, lower_limit_least_sq, max_like_least_sq, args=(Popt2[0], Popt2[1], Popt2[2]))[0]
      lower_limit_least_sq -= 0.1

    lower_prop_least_sq = maxwell_extra(lower_limit_least_sq, *Popt2)

    #Test code to plot distribution
    #plt.figure()
    #plt.plot(Bins_mid_points, N_norm,'x')

    #plt.plot(Bins_mid_plot_points, Probability_curve_fit, color="r", lw=3,marker='')
    #plt.plot(max_like_curve_fit, max_prop_curve_fit, 'ro')
    #plt.vlines(x=upper_limit_curve_fit, color='r', ymin=0., ymax=upper_prop_curve_fit)
    #plt.vlines(x=lower_limit_curve_fit, color='r', ymin=0., ymax=lower_prop_curve_fit)

    #plt.plot(Bins_mid_plot_points, Probability_least_sq, color="b", lw=3,marker='')
    #plt.plot(max_like_least_sq, max_prop_least_sq,'bo')
    #plt.vlines(x=upper_limit_least_sq, color='b', ymin=0., ymax=upper_prop_least_sq)
    #plt.vlines(x=lower_limit_least_sq, color='b', ymin=0., ymax=lower_prop_least_sq)

    #print "Curve", max_like_curve_fit, "+", upper_limit_curve_fit-max_like_curve_fit, "/ -", max_like_curve_fit-lower_limit_curve_fit
    #print "Leastsq", max_like_least_sq, "+", upper_limit_least_sq-max_like_least_sq, "/ -", max_like_least_sq-lower_limit_least_sq

    return max_like_least_sq, lower_limit_least_sq, upper_limit_least_sq

def max_likehood_find_equal(Sample,nbins):
    '''
    This function is like the max_likehood_find function, except that it finds equal
    probability upper and lower limits.
    '''

    N, Bins = np.histogram(Sample,range=(0,np.max(Sample)),bins=nbins,density=False)
    Bins_mid_points = (0.5*(Bins + np.roll(Bins, 1)))[1:]
    N = N / np.max(N)

    N_norm, Bins = np.histogram(Sample,range=(0,np.max(Sample)),bins=nbins,density=True)

    Bins_mid_plot_points = np.linspace(np.min(Bins_mid_points), np.max(Bins_mid_points), 5000)

    #The least squares method is executed here
    Popt2, Pcov2 = leastsq(func=residuals, x0=(0.,1.,300.), args=(Bins_mid_points, N_norm))
    Probability_least_sq  = maxwell_extra(Bins_mid_plot_points, *Popt2)
    max_like_least_sq =  Bins_mid_plot_points[np.argmax(Probability_least_sq)]
    max_prop_least_sq = np.max(Probability_least_sq)

    Prob_least_sq_lower = Probability_least_sq[Bins_mid_plot_points<max_like_least_sq]
    Bins_mid_points_lower = Bins_mid_plot_points[Bins_mid_plot_points<max_like_least_sq]
    f_lower = interp1d(Prob_least_sq_lower, Bins_mid_points_lower)

    Prob_least_sq_upper = Probability_least_sq[Bins_mid_plot_points>max_like_least_sq]
    Bins_mid_points_upper = Bins_mid_plot_points[Bins_mid_plot_points>max_like_least_sq]
    f_upper = interp1d(Prob_least_sq_upper[::-1], Bins_mid_points_upper[::-1])

    prop_sum_least_sq = 0.

    prob_limit_least_sq = max_prop_least_sq * 0.95

    while (prop_sum_least_sq < 0.6827):
        lower_limit_least_sq = f_lower(prob_limit_least_sq)
        upper_limit_least_sq = f_upper(prob_limit_least_sq)
        prop_sum_least_sq = quad(maxwell_extra, lower_limit_least_sq, upper_limit_least_sq, args=(Popt2[0], Popt2[1], Popt2[2]))[0]
        #print "data:", prob_limit_least_sq, prop_sum_least_sq
        prob_limit_least_sq -= 0.005 * max_prop_least_sq 
    #Test code to plot distribution
    '''
    plt.figure()
    plt.plot(Bins_mid_points, N_norm,'x')

    plt.plot(Bins_mid_plot_points, Probability_least_sq, color="b", lw=3,marker='')
    plt.plot(max_like_least_sq, max_prop_least_sq,'bo')

    plt.plot(Bins_mid_plot_points, maxwell(Bins_mid_plot_points,Popt2[2]))
    plt.vlines(x=upper_limit_least_sq, color='b', ymin=0., ymax=prob_limit_least_sq)
    plt.vlines(x=lower_limit_least_sq, color='b', ymin=0., ymax=prob_limit_least_sq)

    print "Leastsq", max_like_least_sq, "+", upper_limit_least_sq-max_like_least_sq, "/ -", max_like_least_sq-lower_limit_least_sq
    plt.show()
    '''
    return max_like_least_sq, lower_limit_least_sq, upper_limit_least_sq



def const_volume():
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "1"
    Angles = np.array([1.0,0.75,0.5,0.37,0.25,0.17,0.125,0.062])

    #n_mle = 1689
    n_mle = 689
    n_mv = 84
    #----------------------------------------------------------------------------------------
    #---------------------------------- !!!SET PARAMETERS !!! -------------------------------
    #----------------------------------------------------------------------------------------

    Means_sum = np.array([])
    Upper_sum = np.array([])
    Lower_sum = np.array([])
    Std_sum = np.array([])
    
    #Means_sum_weight = np.array([])
    #Std_sum_weight = np.array([])

    Means_mle = np.array([])
    Upper_mle = np.array([])
    Lower_mle = np.array([])
    Completeness_mle = np.array([])

    Means_mv = np.array([])
    Upper_mv = np.array([])
    Lower_mv = np.array([])
    Completeness_mv = np.array([])

    #-------------------------------------- MAIN PROGRAM ------------------------------------

    for ang in Angles:
        #print ang
        #print "BulkFlow/" + sub_dir + "/Sum/Bulk_flows_sum_" + str(ang) + ".txt"
        #print "Sum"
        Bulks_sum = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/Sum/Bulk_flows_sum_" + str(ang) + ".txt")
        Bulks_sum_magnitudes = np.sqrt(Bulks_sum[:,0]**2 + Bulks_sum[:,1]**2 + Bulks_sum[:,2]**2)
        mean_sum, upper_sum, lower_sum = max_likehood_find_equal(Bulks_sum_magnitudes,20)
        Means_sum = np.append(Means_sum,mean_sum)
        Upper_sum = np.append(Upper_sum,upper_sum)
        Lower_sum = np.append(Lower_sum,lower_sum)

        #plt.figure()
        #plt.title(str(ang))
        #plt.hist(Bulks_sum,bins=20)

        #Bulks_sum_weight = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/SumWeight/Bulk_flows_sum_" + str(ang) + ".txt")
        #Means_sum_weight = np.append(Means_sum_weight,np.mean(Bulks_sum_weight))
        #Std_sum_weight = np.append(Std_sum_weight,np.std(Bulks_sum_weight))

        for i in np.arange(n_mle):
            Vectors_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  "_n500.txt")
            Bulks_mle = np.sqrt( Vectors_mle[:,0]**2 + Vectors_mle[:,1]**2 + Vectors_mle[:,2]**2 )
            Completeness_mle = np.append(Completeness_mle,Bulks_mle)
        #print "ML"
        mean_mle, upper_mle, lower_mle = max_likehood_find_equal(Completeness_mle,80)
        Means_mle = np.append(Means_mle,mean_mle)
        Upper_mle = np.append(Upper_mle,upper_mle)
        Lower_mle = np.append(Lower_mle,lower_mle)
        Completeness_mle = np.array([])
        
        for i in np.arange(n_mv):
            Vectors_mv = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MV/MV_Bulk_flows_" + str(ang) + "_n300_" + str(i) +  ".txt")
            Bulks_mv = np.sqrt( Vectors_mv[:,0]**2 + Vectors_mv[:,1]**2 + Vectors_mv[:,2]**2 )
            Completeness_mv = np.append(Completeness_mv,Bulks_mv)
        mean_mv, upper_mv, lower_mv = max_likehood_find_equal(Completeness_mv,80)
        Means_mv = np.append(Means_mv,mean_mv)
        Upper_mv = np.append(Upper_mv,upper_mv)
        Lower_mv = np.append(Lower_mv,lower_mv)
        Completeness_mv = np.array([])
        


    #Read in theoretical estimate
    Angles_theory = np.array([360., 270., 180., 135., 90., 60., 45., 22.5])
    
    #Results from FFTW
    V_var_theory = np.array([141.029, 136.608, 134.455, 134.247, 136.394, 137.609, 135.635, 130.148])
    #V_var_theory = np.array([80.0837, 80.7018, 85.2883, 85.956, 84.6302, 80.1531, 74.8101, 61.9187])

    #Results from Vegas
    #V_var_theory = np.array([138.357539546, 134.175504012, 129.06311408, 127.183116498, 131.193476721, 134.561244168, 130.533631684, 113.456157539])
    
    V_max_theory =  V_var_theory * np.sqrt(2./3.)
    
    V_theory_upper = V_max_theory + 0.419 * V_var_theory
    V_theory_lower = V_max_theory - 0.356 * V_var_theory
    
    #---------------------------------------- PLOTTING -------------------------------------
    
    plt.figure()
    #plt.title("Cosmic variance for constant-volume geometries")
    
    
    Angles = Angles * 360.0
    
    if True:
        
        plt.plot(Angles,Upper_sum,ls='-',marker='',lw=3,color='r',alpha=0.5)
        plt.plot(Angles,Means_sum,ls='--',marker='',lw=3,color='r',alpha=0.4,label='Simulation')
        plt.plot(Angles,Lower_sum,ls='-',marker='',lw=3,color='r',alpha=0.5)

    if True:
        plt.fill_between(Angles_theory,V_theory_upper,V_theory_lower,color='g',alpha=0.1)
        plt.plot(Angles_theory,V_theory_upper,ls='-',lw=3,color='g',alpha=0.7)
        plt.plot(Angles_theory,V_max_theory,ls='--',lw=3,color='g',alpha=0.7,label='Theory')
        plt.plot(Angles_theory,V_theory_lower,ls='-',lw=3,color='g',alpha=0.7) 
    
    if True:
        plt.fill_between(Angles,Upper_mv,Lower_mv,color='darkorange',alpha=0.1)
        plt.plot(Angles,Upper_mv,ls='-',marker='',lw=3,color='darkorange',alpha=0.5)
        plt.plot(Angles,Means_mv,ls='--',marker='',lw=3,color='darkorange',alpha=0.4,label='MV', dashes=(12,4))
        plt.plot(Angles,Lower_mv,ls='-',marker='',lw=3,color='darkorange',alpha=0.5)
    
    if True:
        plt.fill_between(Angles,Upper_mle,Lower_mle,color='blue',alpha=0.1)
        plt.plot(Angles,Upper_mle,ls='-',marker='',lw=3,color='darkblue',alpha=0.5)
        plt.plot(Angles,Means_mle,ls='--',marker='',lw=3,color='darkblue',alpha=0.4,label='ML', dashes=(12,4))
        plt.plot(Angles,Lower_mle,ls='-',marker='',lw=3,color='darkblue',alpha=0.5)
    
    #plt.plot(Angles,Means_sum_weight+Std_sum_weight,ls='-',marker='',lw=3,color='grey',label='Velocity Sum Weighted',alpha=0.5)
    #plt.plot(Angles,Means_sum_weight,ls='--',marker='',lw=3,color='grey',alpha=0.2)
    #plt.plot(Angles,Means_sum_weight-Std_sum_weight,ls='-',marker='',lw=3,color='grey',alpha=0.5)

    #print Means_sum
    legend_plot(fsize=22)
    plt.xlim((0,360))
    plt.ylim((0,450))
    plt.xlabel(r" Opening Angle $\mathrm{\theta}$",size='xx-large')
    plt.ylabel(r"Most Probable Bulk Flow [km$\,$s$^{-1}$]",size='xx-large')
    plt.xticks([0, 180, 360], [r'0', r'$\mathrm{\pi/2}$', r'$\mathrm{\pi}$'],size='xx-large')
    plt.yticks([0, 150, 300, 450],size='xx-large')
    plt.subplots_adjust(left=0.13, bottom=0.13)
    plt.show()
    
def const_volume_poster():
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
    Upper_sum = np.array([])
    Lower_sum = np.array([])
    Std_sum = np.array([])
    
    #Means_sum_weight = np.array([])
    #Std_sum_weight = np.array([])

    Means_mle = np.array([])
    Upper_mle = np.array([])
    Lower_mle = np.array([])
    Completeness_mle = np.array([])

    Means_mv = np.array([])
    Upper_mv = np.array([])
    Lower_mv = np.array([])
    Completeness_mv = np.array([])

    #-------------------------------------- MAIN PROGRAM ------------------------------------

    for ang in Angles:
      Bulks_sum = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/Sum/Bulk_flows_sum_" + str(ang) + ".txt")
      mean_sum, upper_sum, lower_sum = max_likehood_find(Bulks_sum,20)
      Means_sum = np.append(Means_sum,mean_sum)
      Upper_sum = np.append(Upper_sum,upper_sum)
      Lower_sum = np.append(Lower_sum,lower_sum)
      
      #plt.figure()
      #plt.title(str(ang))
      #plt.hist(Bulks_sum,bins=20)
      
      #Bulks_sum_weight = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/SumWeight/Bulk_flows_sum_" + str(ang) + ".txt")
      #Means_sum_weight = np.append(Means_sum_weight,np.mean(Bulks_sum_weight))
      #Std_sum_weight = np.append(Std_sum_weight,np.std(Bulks_sum_weight))
  
      for i in np.arange(n_mle):
        Vectors_mle = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" + str(ang) + "_" + str(i) +  ".txt")
        Bulks_mle = np.sqrt( Vectors_mle[:,0]**2 + Vectors_mle[:,1]**2 + Vectors_mle[:,2]**2 )
        Completeness_mle = np.append(Completeness_mle,Bulks_mle)
      
      mean_mle, upper_mle, lower_mle = max_likehood_find(Completeness_mle,80)
      Means_mle = np.append(Means_mle,mean_mle)
      Upper_mle = np.append(Upper_mle,upper_mle)
      Lower_mle = np.append(Lower_mle,lower_mle)
      Completeness_mle = np.array([])
  
      Completeness_mv = np.array([])
    
    #---------------------------------------- PLOTTING -------------------------------------
    
    plt.figure(figsize=(17,6))
    #plt.title("Cosmic variance for constant-volume geometries")
    plt.xlim((0,360))
    plt.xticks([0,45,90,135,180,225,270,315,360])
    plt.ylim((0,450))
    plt.xlabel(r"Angle - $\theta$",size='xx-large',weight='bold')
    plt.ylabel(r"Bulk Flow Magnitude - [km$\,$s$^{-1}$]",size='xx-large',weight='bold')
    
    Angles = Angles * 360.0

    plt.fill_between(Angles,Upper_mle,Lower_mle,color='blue',alpha=0.3)
    plt.fill_between(Angles,Upper_sum,Lower_sum,color='r',alpha=0.3)
    
    plt.plot(Angles,Upper_mle,ls='-',marker='',lw=3,color='darkblue',label='Maximum Likelihood',alpha=0.5)
    plt.plot(Angles,Means_mle,ls='--',marker='',lw=3,color='darkblue',alpha=0.9)
    plt.plot(Angles,Lower_mle,ls='-',marker='',lw=3,color='darkblue',alpha=0.5)

    plt.plot(Angles,Upper_sum,ls='-',marker='',lw=3,color='r',label='Actual Velocity',alpha=0.5)
    plt.plot(Angles,Means_sum,ls='--',marker='',lw=3,color='r',alpha=0.9)
    plt.plot(Angles,Lower_sum,ls='-',marker='',lw=3,color='r',alpha=0.5)
    
    #plt.plot(Angles,Means_sum_weight+Std_sum_weight,ls='-',marker='',lw=3,color='grey',label='Velocity Sum Weighted',alpha=0.5)
    #plt.plot(Angles,Means_sum_weight,ls='--',marker='',lw=3,color='grey',alpha=0.2)
    #plt.plot(Angles,Means_sum_weight-Std_sum_weight,ls='-',marker='',lw=3,color='grey',alpha=0.5)

    print Means_sum
    legend_plot(fsize=30)
    plt.xticks((0, 90, 180, 270, 360), (r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'), color='k', size=20)
    plt.yticks((0, 100, 200, 300, 400, 450), ('0', '100', '200', '300', '400', '450'), color='k', size=20)
    plt.subplots_adjust(bottom=0.14)
    plt.subplots_adjust(left=0.08)
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
  
      
      #plt.title("HR2 Sampling Variance")
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
    
        
        #plt.title("HR2 Completeness Variance")
        plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
        plt.ylabel(r"Normalised Probability",size='xx-large')
        plt.xlim((0,500))
        plt.hist(Completeness,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
        plt.hist(Completeness,range=(0,500),bins=20,normed=True,histtype='stepfilled',fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)

def cosmic025(angle_string = '0.25'):
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "8A"
    
    #N_per_mle = [1000,500,100,50]
    #N_per_mle = [4000, 2000]#, 1000, 500, 100, 50]
    N_per_mle = [4000, 2000, 1000, 500, 100, 50]
    n_rot = 80
    n_bulk = 2000
    Bulks = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    Colors = ['white','lightgrey','darkgrey','grey','dimgrey','#3F3F3F']
    
    for i in np.arange(n_rot):
        for j in np.arange(len(N_per_mle)):
            Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" +angle_string + "_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
            '''
            print i, N_per_mle[j]
            plt.figure()
            plt.hist(Data[:,0],range=(-150,150),bins=50)
            plt.xlim((-150,150))
            plt.figure()
            plt.hist(Data[:,1],range=(-150,150),bins=50)
            plt.xlim((-150,150))
            plt.figure()
            plt.hist(Data[:,2],range=(-150,150),bins=50)
            plt.xlim((-150,150))
            plt.show()
            '''
            Bulks[i,j,:] = np.sqrt(Data[:,0]**2 + Data[:,1]**2 + Data[:,2]**2)
    
    if (angle_string == '1.0'):
        vel_disp = 135.9464646 * np.sqrt(2./3.)
        #vel_disp = 59.697 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
        vel = vel_disp * np.sqrt(2./3.)
    
    elif (angle_string == '0.5'):
        vel_disp = 126.7737192 * np.sqrt(2./3.)
        #vel_disp = 70.1976 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
        vel = vel_disp * np.sqrt(2./3.)
    
    elif (angle_string == '0.125'):
        vel_disp = 128.5138262 * np.sqrt(2./3.)
        #vel_disp = 142.143 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
        vel = vel_disp * np.sqrt(2./3.)
        
    else: #0.25
        vel_disp = 129.0859657 * np.sqrt(2./3.)
        #vel_disp = 100.187 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
        vel = vel_disp * np.sqrt(2./3.)
        
    
    plt.figure()
    plt.title('Opening angle:' + angle_string)
    print "HR2 MLE Cosmic Variance"
    for i in np.arange(len(N_per_mle)):
        Completeness = Bulks[:,i,:]
        Completeness = Completeness.reshape((n_rot*n_bulk))
        
        print "n: ", N_per_mle[i], "->",np.mean(Completeness), "+/-", np.std(Completeness)
    
        
        #plt.title("HR2 Cosmic Variance")
        plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
        plt.ylabel(r"Normalised Probability",size='xx-large')
        plt.xlim((0,500))
        plt.hist(Completeness,range=(0,500),bins=60,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
        plt.hist(Completeness,range=(0,500),bins=60,normed=True,histtype='stepfilled',fill=False)
    plt.axvline(vel,color='r',lw = 2.)
    plt.axvspan(vel - 0.356*vel_disp, vel + 0.419*vel_disp,alpha=0.4,color='r',label="LinTheo")
    plt.axvspan(vel - 0.619*vel_disp, vel + 0.891*vel_disp,alpha=0.3,color='r')
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=25)

def cosmic_combined():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "8A"
    
    #N_per_mle = [1000,500,100,50]
    #N_per_mle = [8000, 4000, 2000]#, 1000, 500, 100, 50]
    N_per_mle = [4000, 2000, 1000, 500, 50]
    #N_per_mle = [4000, 2000, 1000, 500, 100, 50]
    n_rot = 1689
    n_bulk = 5
    Bulks = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    Colors = ['white','lightgrey','darkgrey','grey','dimgrey','#3F3F3F']
    
    plot_i = 1
    f = plt.figure(figsize=(6,10))
    for angle_string in ['0.125', '0.5', '1.0']:
        if (plot_i == 1):
            ax1 = plt.subplot(311)
        if (plot_i == 2):
            ax2 = plt.subplot(312, sharex=ax1)
        if (plot_i == 3):
            ax3 = plt.subplot(313, sharex=ax1)
        for i in np.arange(n_rot):
            for j in np.arange(len(N_per_mle)):
                data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" +angle_string + "_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
                #print angle_string, N_per_mle[j]
                #plt.figure()
                #plt.hist(Data[:,0], bins = 20)
                #plt.figure()
                #plt.hist(Data[:,1], bins = 20)
                #plt.figure()
                #plt.hist(Data[:,2], bins = 20)
                #plt.show()
                #Data = np.genfromtxt(root_dir + "BulkFlow/8A/MLE/MLE_Bulk_flows_1.0_" + str(i) + "_4000.txt")
                '''
                print i, N_per_mle[j]
                plt.figure()
                plt.hist(Data[:,0],range=(-150,150),bins=50)
                plt.xlim((-150,150))
                plt.figure()
                plt.hist(Data[:,1],range=(-150,150),bins=50)
                plt.xlim((-150,150))
                plt.figure()
                plt.hist(Data[:,2],range=(-150,150),bins=50)
                plt.xlim((-150,150))
                plt.show()
                '''
                #a = np.random.uniform() * np.pi
                #b = np.random.uniform() * np.pi
                #c = np.random.uniform() * np.pi
                #vx_rot = np.cos(b)*np.cos(c)*data[:,0] - np.cos(b)*np.sin(c)*data[:,1] + np.sin(b)*data[:,2]
                #vy_rot = np.sin(a)*np.sin(b)*np.cos(c)*data[:,0] + np.cos(a)*np.sin(c)*data[:,0] - np.sin(a)*np.sin(b)*np.sin(c)*data[:,1] + np.cos(a)*np.cos(c)*data[:,1] - np.sin(a)*np.cos(b)*data[:,2]
                #vz_rot = -np.cos(a)*np.sin(b)*np.cos(c)*data[:,0] + np.sin(a)*np.sin(c)*data[:,0] + np.cos(a)*np.sin(b)*np.sin(c)*data[:,1] + np.sin(a)*np.cos(c)*data[:,1] + np.cos(a)*np.cos(b)*data[:,2]
                #Bulks[i,j,:] = np.sqrt(vx_rot**2 + vy_rot**2 + vz_rot**2)
                Bulks[i,j,:] = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)

        if (angle_string == '1.0'):
            vel_disp = 141.029
            #vel_disp = 59.697 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
    
        elif (angle_string == '0.5'):
            vel_disp = 134.455
            #vel_disp = 70.1976 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
    
        elif (angle_string == '0.125'):
            vel_disp = 135.635
            #vel_disp = 142.143 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
        
        else: #0.25
            vel_disp = 136.394
            #vel_disp = 100.187 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
        
        #plt.title('Opening angle:' + angle_string)
        print "HR2 MLE Cosmic Variance"
        plt.axvline(vel,color='r',lw = 2.)
        plt.axvspan(vel - 0.356*vel_disp, vel + 0.419*vel_disp,alpha=0.4,color='r',label="LinTheo")
        plt.axvspan(vel - 0.619*vel_disp, vel + 0.891*vel_disp,alpha=0.3,color='r')
        for i in np.arange(len(N_per_mle)):
            Completeness = Bulks[:,i,:]
            Completeness = Completeness.reshape((n_rot*n_bulk))
        
            print "n: ", N_per_mle[i], "->",np.mean(Completeness), "+/-", np.std(Completeness)
    
        
            #plt.title("HR2 Cosmic Variance")
            if (plot_i == 1):
                if N_per_mle[i] == N_per_mle[0]:
                    ax1.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False, ls='dotted',label=r"$n$ : " + str(N_per_mle[i]))
                    ax1.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],alpha=1.)
                elif N_per_mle[i] == N_per_mle[-1]:
                    ax1.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],alpha=1., label=r"$n$ : " + str(N_per_mle[i]))
                    ax1.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
                else:
                    ax1.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
                    ax1.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],alpha=0.6,label=r"$n$ : " + str(N_per_mle[i]))
                ax1.set_yticks([0.0, 0.002, 0.004, 0.006, 0.008])
                ax1.set_ylim([0., 0.008])
            if (plot_i == 2):
                if N_per_mle[i] == N_per_mle[0]:
                    ax2.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False, ls='dotted',label=r"$n$ : " + str(N_per_mle[i]))
                    ax2.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],alpha=1.)
                elif N_per_mle[i] == N_per_mle[-1]:
                    ax2.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],alpha=1., label=r"$n$ : " + str(N_per_mle[i]))
                    ax2.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
                else:
                    ax2.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
                    ax2.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],alpha=0.6,label=r"$n$ : " + str(N_per_mle[i]))
                ax2.set_yticks([0.0, 0.002, 0.004, 0.006, 0.008])
                ax2.set_ylim([0., 0.008])
            if (plot_i == 3):
                if N_per_mle[i] == N_per_mle[0]:
                    ax3.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False, ls='dotted',label=r"$n$ : " + str(N_per_mle[i]))
                    ax3.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],alpha=1.)
                elif N_per_mle[i] == N_per_mle[-1]:
                    ax3.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],alpha=1., label=r"$n$ : " + str(N_per_mle[i]))
                    ax3.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
                else:
                    ax3.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
                    ax3.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],alpha=0.6,label=r"$n$ : " + str(N_per_mle[i]))
                ax3.set_yticks([0.0, 0.002, 0.004, 0.006, 0.008])
                ax3.set_ylim([0., 0.008])
        plt.xlim((0,600))
        #plt.ylim((0,0.015))
        
        
        if (plot_i == 3):
            legend_plot(fsize=16, location=7)
        plot_i += 1
    
    plt.figtext(0.74, 0.95, r'$\theta$ : $\pi/8$',size='xx-large')
    plt.figtext(0.74, 0.65, r'$\theta$ : $\pi/2$',size='xx-large')
    plt.figtext(0.74, 0.35, r'$\theta$ : $\pi$',size='xx-large')
    plt.subplots_adjust(left=0.14)
    plt.subplots_adjust(bottom=0.07)
    plt.subplots_adjust(right=0.95)
    plt.subplots_adjust(top=0.99)
    plt.subplots_adjust(hspace=0.001)
    xticklabels = ax1.get_xticklabels() + ax2.get_xticklabels()
    plt.setp(xticklabels, visible=False)
    plt.xlabel(r"Most Probable Bulk Flow [km$\,$s$^{-1}$]",size='xx-large')
    #plt.ylabel(r"Normalised Probability",size='x-large')

def cosmic_combined_horizontal():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "8A"

    #N_per_mle = [4000, 2000, 1000, 500, 100, 50]
    N_per_mle = [4000, 500, 50]
    n_rot = 1689
    n_bulk = 5
    Bulks = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    Colors = ['darkgrey','grey','k']
    
    plot_i = 1
    f = plt.figure(figsize=(13,6))
    for angle_string in ['0.125', '0.5', '1.0']:
        if (plot_i == 1):
            ax1 = plt.subplot(131)
        if (plot_i == 2):
            ax2 = plt.subplot(132)
        if (plot_i == 3):
            ax3 = plt.subplot(133)
        for i in np.arange(n_rot):
            for j in np.arange(len(N_per_mle)):
                data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" +angle_string + "_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
                Bulks[i,j,:] = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)

        if (angle_string == '1.0'):
            vel_disp = 141.029
            #vel_disp = 59.697 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
    
        elif (angle_string == '0.5'):
            vel_disp = 134.455
            #vel_disp = 70.1976 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
    
        elif (angle_string == '0.125'):
            vel_disp = 135.635
            #vel_disp = 142.143 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
        
        else: #0.25
            vel_disp = 136.394
            #vel_disp = 100.187 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
        
        plt.axvline(vel,color='r',lw = 2.)
        plt.axvspan(vel - 0.356*vel_disp, vel + 0.419*vel_disp,alpha=0.4,color='r',label="LinTheo")
        plt.axvspan(vel - 0.619*vel_disp, vel + 0.891*vel_disp,alpha=0.3,color='r')
        #plt.title('Opening angle:' + angle_string)
        print "HR2 MLE Cosmic Variance"
        for i in np.arange(len(N_per_mle)):
            Completeness = Bulks[:,i,:]
            Completeness = Completeness.reshape((n_rot*n_bulk))
        
            print "n: ", N_per_mle[i], "->",np.mean(Completeness), "+/-", np.std(Completeness)
    
        
            #plt.title("HR2 Cosmic Variance")
            if (plot_i == 1):
                ax1.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.8)
                ax1.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
                ax1.set_yticks([0.0, 0.002, 0.004, 0.006, 0.008])
                ax1.set_ylim([0., 0.008])
                plt.setp(ax1.get_yticklabels(), fontsize=14)
                ax1.set_xticks([0, 200, 400])
                plt.setp(ax1.get_xticklabels(), fontsize=14)

                plt.ylabel(r"Normalised Probability",size='xx-large')
            if (plot_i == 2):
                ax2.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.8)
                ax2.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
                ax2.set_ylim([0., 0.008])
                ax2.set_xticks([0, 200, 400])
                plt.setp(ax2.get_xticklabels(), fontsize=14)
                plt.xlabel(r"Most Probable Bulk Flow [km$\,$s$^{-1}$]",size='xx-large')
            if (plot_i == 3):
            
                ax3.hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.8)
                ax3.hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
                ax3.set_ylim([0., 0.008])
                ax3.set_xticks([0, 200, 400, 600])
                plt.setp(ax3.get_xticklabels(), fontsize=14)
        #plt.xlim((0,600))
        #plt.ylim((0,0.015))
        #plt.axvline(vel,color='r',lw = 2.)
        #plt.axvspan(vel - 0.356*vel_disp, vel + 0.419*vel_disp,alpha=0.4,color='r',label="LinTheo")
        #plt.axvspan(vel - 0.619*vel_disp, vel + 0.891*vel_disp,alpha=0.3,color='r')
        
        if (plot_i == 3):
            legend_plot(fsize=16, location=1)
        plot_i += 1
    
    plt.figtext(0.7, 0.91, r'$\theta$ : $\pi$, Sphere',size='xx-large')
    plt.figtext(0.4, 0.91, r'$\theta$ : $\pi/2$, Hemisphere',size='xx-large')
    plt.figtext(0.13, 0.91, r'$\theta$ : $\pi/8$, Cone',size='xx-large')
    plt.subplots_adjust(left=0.10, bottom=0.12, right=0.97, top=0.96, wspace=0.0, hspace=0.0)
    yticklabels = ax2.get_yticklabels() + ax3.get_yticklabels()
    plt.setp(yticklabels, visible=False)

def cosmic_print_values():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "8A"
    
    N_per_mle = [8000, 4000, 2000, 1000, 500, 100, 50]

    n_rot = 1689
    n_bulk = 5
    Bulks = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    latex_string = "\\begin{table*} \n \\centering \n \\begin{tabular}{| c | c | c | c | c | c |} \n \hline $ $ & $\\theta$ & V$_p$ & 68\\% Limits&$\\mid$V$_p$ - V$_{p,\\mathrm{theory}}\\mid$ & Sample Density\\\\"
    latex_string += "\n $ $ & $ $ & km s$^{-1}$ & km s$^{-1}$ & km s$^{-1}$ & $(h^{-1}\\mathrm{Mpc})^{-3}$\\\\ \\hline"

    Angles = np.array([1.0,0.5,0.37,0.25,0.17,0.125,0.062])
    #Results from FFTW
    #V_var_theory = np.array([141.029, 136.608, 134.455, 134.247, 136.394, 137.609, 135.635, 130.148])
    #Results from Vegas
    #V_var_theory = np.array([138.358, 134.176, 129.06311408, 127.183116498, 131.193476721, 134.561244168, 130.533631684, 113.456157539])
    
    numbers_column = 3*N_per_mle
    numbers_column = [("n : " + str(ii)) for ii in numbers_column]
    angle_column = []
    prob_bulk_flow_column = []
    one_sigma_column = []
    relative_shift_column = []
    delta_bulk = []
    sample_density_column = []
    
    for angle_string in ['0.125', '0.5', '1.0']:
        for i in np.arange(n_rot):
            for j in np.arange(len(N_per_mle)):
                data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_" +angle_string + "_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
                Bulks[i,j,:] = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)

        if (angle_string == '1.0'):
            vel_disp = 138.358 #Vegas
            #vel_disp = 141.029 #FFTW
            #vel_disp = 59.697 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
    
        elif (angle_string == '0.5'):
            vel_disp = 134.176
            #vel_disp = 70.1976 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
    
        elif (angle_string == '0.125'):
            vel_disp = 130.533631684
            #vel_disp = 142.143 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
        
        else: #0.25
            vel_disp = 127.183
            #vel_disp = 100.187 * np.sqrt(2./3.) #To be used for radius 690 Mpc/h
            vel = vel_disp * np.sqrt(2./3.)
        
        print "HR2 MLE Cosmic Variance"
        print "Angle:", angle_string, "v_theory:", vel
        for i in np.arange(len(N_per_mle)):
            angle_column.append(angle_string + " $ \\, \\pi$")
            Completeness = Bulks[:,i,:]
            Completeness = Completeness.reshape((n_rot*n_bulk))
            max_like_least_sq, lower_limit_least_sq, upper_limit_least_sq = max_likehood_find_equal(Completeness,60)
            prob_bulk_flow_column.append(str(int(np.round(max_like_least_sq,0))) + " $^{+" + str(int(np.round(upper_limit_least_sq - max_like_least_sq,0))) + "}_{-" + str(int(np.round(max_like_least_sq - lower_limit_least_sq,0))) + "}$")
            one_sigma_column.append(str(int(lower_limit_least_sq)) + ' - ' + str(int(upper_limit_least_sq)))
            relative_shift_column.append(str(int(np.abs(np.round(1. - vel / max_like_least_sq,2))*100.)) + " \\%" )
            sample_density_column.append(str(int(np.round(N_per_mle[i] / 40.,1))) + "$ \\times 10^{-6}$")
            #delta_bulk.append(str(int(np.abs(np.round((vel - max_like_least_sq)/10.,0)*10.))) + " km s$^{-1}$")
            delta_bulk.append(str(int(np.abs(np.round((vel - max_like_least_sq),0)))))
            print "n :", N_per_mle[i], "v_p : " + str(int(np.round(max_like_least_sq,0))) + " +" + str(int(np.round(upper_limit_least_sq - max_like_least_sq,0))) + "/-" + str(int(np.round(max_like_least_sq - lower_limit_least_sq,0))) + ", "+ str(int(np.abs(np.round(1. - vel / max_like_least_sq,2))*100.)) + "% offset"
            '''
            plt.figure()
            plt.hist(Completeness, bins=30)
            plt.axvline(max_like_least_sq, color='r')
            plt.axvline(upper_limit_least_sq, color='r')
            plt.axvline(lower_limit_least_sq, color='r')
            plt.show()
            '''
    #print numbers_column
    #print angle_column
    #print prob_bulk_flow_column
    #print relative_shift_column
    #print sample_density_column
    for ii in np.arange(len(N_per_mle)):
    	latex_string += "\n" + numbers_column[ii] + " & "
        latex_string += angle_column[ii] + " & "
        latex_string += prob_bulk_flow_column[ii] + " & "
        latex_string += one_sigma_column[ii] + " & "
        #latex_string += relative_shift_column[ii] + " & "
        latex_string += delta_bulk[ii] + " & "
        latex_string += sample_density_column[ii] + " \\\\ "
    latex_string += "\\hline \\hline"
    for ii in np.arange(len(N_per_mle)):
        ii += len(N_per_mle)
        latex_string += "\n" + numbers_column[ii] + " & "
        latex_string += angle_column[ii] + " & "
        latex_string += prob_bulk_flow_column[ii] + " & "
        latex_string += one_sigma_column[ii] + " & "
        #latex_string += relative_shift_column[ii] + " & "
        latex_string += delta_bulk[ii] + " & "
        latex_string += sample_density_column[ii] + " \\\\ "
    latex_string += "\\hline \\hline"
    for ii in np.arange(len(N_per_mle)):
        ii += 2*len(N_per_mle)
        latex_string += "\n" + numbers_column[ii] + " & "
        latex_string += angle_column[ii] + " & "
        latex_string += prob_bulk_flow_column[ii] + " & "
        latex_string += one_sigma_column[ii] + " & "
        #latex_string += relative_shift_column[ii] + " & "
        latex_string += delta_bulk[ii] + " & "
        latex_string += sample_density_column[ii] + " \\\\ "
    latex_string += " \\hline\n \\end{tabular} \n"
    latex_string += "\\caption{$V_p$ is the most probable bulk flow for the distribution of bulk flows derived from simulation using the ML estimator, for the given survey geometry, defined by the opening angle $\\theta$, and sampling rate, given by $n$, which is the number of peculiar velocities per derived bulk flow estimate. The upper and lower equal likelihood limits encapsulating 68\\% of the likelihood are also listed. $\mid$V$_p$ - V$_{p,\mathrm{th}}\mid$ is the absolute difference between the most probable bulk flow velocity derived from estimate and from linear theory. A small absolute difference indicates that the sampling rate is sufficient for the given geometry, such that the derived distribution matches the actual underlying distribution. In the final column the survey sample density in samples per million $(h^{-1}\\mathrm{Mpc})^3$ is listed for reference.} \n"
    latex_string += "\\label{tab:samplinggeometryeffectscombined} \n"
    latex_string += "\\end{table*}" 
    print ""

    print latex_string

def cosmic_gaussians():
	root_dir = "/Users/perandersen/Data/"
	sub_dir = "8A"
	#N_per_mle = [8000, 4000, 2000]#, 1000, 500, 100, 50]
	N_per_mle = [4000, 2000, 1000, 500, 100, 50]
	n_rot = 1689
	n_bulk = 5

	Colors = ['white','lightgrey','darkgrey','grey','dimgrey','#3F3F3F']
	plot_i = 1
	
	Bulks_x = np.zeros( (len(N_per_mle), n_rot, n_bulk) )

	for ii in np.arange(n_rot):
		for jj in np.arange(len(N_per_mle)):
			data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.125_" + str(ii) + "_" + str(N_per_mle[jj]) + ".txt")
			Bulks_x[jj, ii, :] = data[:,0]
	print np.shape(Bulks_x)
	Bulks_x = Bulks_x.reshape(len(N_per_mle), n_rot*n_bulk)
	print np.shape(Bulks_x)
	plt.figure()
	for ii in np.arange(len(N_per_mle)):
		#hist(Completeness,range=(0,600),bins=24,normed=True,edgecolor='none',color=Colors[i],label=r"$n$ : " + str(N_per_mle[i]),alpha=0.7)
       	#hist(Completeness,range=(0,600),bins=24,normed=True,histtype='stepfilled',fill=False)
		plt.hist(Bulks_x[ii,:], bins=50, normed=True, color=Colors[ii], range=(-1000,1000), edgecolor='none', alpha=0.7,label=r"$n$ : " + str(N_per_mle[ii]))
		plt.hist(Bulks_x[ii,:], bins=50, normed=True, range=(-1000,1000), alpha=0.7, histtype='stepfilled', fill=False)
	legend_plot(fsize=22)
	plt.xticks([-1000, -500, 0, 500, 1000], size='xx-large')
	plt.yticks([0.001, 0.002, 0.003, 0.004], size='xx-large')
	plt.xlabel(r"Bulk Flow Vector X-Component [km$\,$s$^{-1}$]",size='xx-large')
	plt.subplots_adjust(bottom=0.15)

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
  
      
      #plt.title("HR2 Sampling Variance - FoF DM Halo")
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
    N_per_mle = [50,100,500]
    Colors_mock = 'darksalmon','palegreen','deepskyblue','dimgrey','#3F3F3F'
    Colors_fof = 'firebrick','darkgreen','b','dimgrey','#3F3F3F'
    #Colors_fof = ['dimgrey','darkgrey','lightgrey']
    #Colors_mock = ['dimgrey','darkgrey','lightgrey']
    
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
      
      #plt.title("HR2 Sampling Variance - SDSSIII Mock and FoF DM Halo")
      plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
      plt.ylabel(r"Normalised Probability",size='xx-large')
      plt.xlim((0,500))
      plt.hist(Bulks_mle_mock,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors_mock[i],label=r"$n$ : " + str(N_per_mle[i]) + ", Mock",alpha=0.6)
      plt.hist(Bulks_mle_mock,range=(0,500),bins=20,normed=True,histtype='stepfilled',edgecolor=Colors_mock[i],fill=False)
      plt.hist(Bulks_mle_fof,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors_fof[i],label=r"$n$ : " + str(N_per_mle[i]) + ", DM Halo",alpha=0.6)
      plt.hist(Bulks_mle_fof,range=(0,500),bins=20,normed=True,histtype='stepfilled',linestyle=('dashed'),edgecolor=Colors_fof[i],fill=False)
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=23)
def comp025_FoF_mock():
    root_dir = "/Users/perandersen/Data/"
    sub_dir_mock = "9"
    sub_dir_fof = "11"
    
    #N_per_mle = [50,100,500,2000]
    N_per_mle = [50,100,500]
    Colors_mock = 'darksalmon','palegreen','deepskyblue','dimgrey','#3F3F3F'
    Colors_fof = 'firebrick','darkgreen','b','dimgrey','#3F3F3F'
    
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
    
        
        
        plt.xlim((0,500))
        plt.hist(Completeness_mock,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors_mock[i],label=r"$n$ : " + str(N_per_mle[i]) + ", Mock",alpha=0.6)
        plt.hist(Completeness_mock,range=(0,500),bins=20,normed=True,histtype='stepfilled',edgecolor=Colors_mock[i],fill=False)
        plt.hist(Completeness_fof,range=(0,500),bins=20,normed=True,edgecolor='none',color=Colors_fof[i],label=r"$n$ : " + str(N_per_mle[i]) + ", DM Halo",alpha=0.6)
        plt.hist(Completeness_fof,range=(0,500),bins=20,normed=True,histtype='stepfilled',linestyle=('dashed'),edgecolor=Colors_fof[i],fill=False)
        
        most_probable_velocity_mock, upper_velocity_mock, lower_velocity_mock = max_likehood_find_equal(Completeness_mock,20)
        most_probable_velocity_fof, upper_velocity_fof, lower_velocity_fof = max_likehood_find_equal(Completeness_fof,20)
        
        print "n : ", N_per_mle[i]
        print "Mock : ", most_probable_velocity_mock, "+/-", lower_velocity_mock - most_probable_velocity_mock, "/", most_probable_velocity_mock - upper_velocity_mock
        print "FoF : ", most_probable_velocity_fof, "+/-", lower_velocity_fof - most_probable_velocity_fof, "/", most_probable_velocity_fof - upper_velocity_fof
        print " "
    #plt.title("HR2 Completeness Variance - SDSSIII Mock and FoF DM Halo")
    plt.xlabel(r"Most Probable Bulk Flow [km$\,$s$^{-1}$]",size='xx-large')
    #plt.ylabel(r"Normalised Probability",size='xx-large')
    plt.xticks([0, 100, 200, 300, 400, 500], size='xx-large')
    plt.yticks([0, 0.003, 0.006, 0.009], size='xx-large')
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=23)
    
    
def cosmic025_maxwell():
    root_dir = "/Users/perandersen/Data/"
    sub_dir = "8"
    
    #N_per_mle = [1000,500,100,50]
    N_per_mle = [2000]
    n_rot = 80
    n_bulk = 2000
    Bulks = np.zeros(  (n_rot,len(N_per_mle),n_bulk)  )
    #Colors = ['grey','dimgrey','#3F3F3F']
    Colors = ['r','g','b','c']
    
    for i in np.arange(n_rot):
        for j in np.arange(len(N_per_mle)):
            Data = np.genfromtxt(root_dir + "BulkFlow/" + sub_dir + "/MLE/MLE_Bulk_flows_0.5_" + str(i) + "_" + str(N_per_mle[j]) + ".txt")
            Bulks[i,j,:] = np.sqrt(Data[:,0]**2 + Data[:,1]**2 + Data[:,2]**2)
    
    plt.figure()
    print "HR2 MLE Cosmic Variance"
    for i in np.arange(len(N_per_mle)):
        Completeness = Bulks[:,i,:]
        Completeness = Completeness.reshape((n_rot*n_bulk))
        
        print "n: ", N_per_mle[i], "->",np.mean(Completeness), "+/-", np.std(Completeness)
    
        #First we set parameters
        nbins = 100
        xmin = 0
        xmax = 200
        plt.xlabel(r"Bulk Flow Velocity [km/s]",size='xx-large')
        
        #Then we find the best-fitting maxwell function
        N, Bins = np.histogram(Completeness,range=(xmin,xmax),bins=nbins,density=True)
        Bins_mid_points = (0.5*(Bins + np.roll(Bins, 1)))[1:]
        Sigma = 1. / np.sqrt(N)
        Popt, Pcov = curve_fit(maxwell, xdata=Bins_mid_points[:], ydata=N[:],p0=(100), sigma=Sigma)
        
        print "Sigma Optimal:", Popt
        #We then plot those results
        plt.subplot(2,1,1)
        plt.xlim((xmin,xmax))
        plt.ylabel(r"Normalised Probability",size='x-large')
        plt.plot(Bins_mid_points,N,'x',color=Colors[i])
        plt.plot(Bins_mid_points, maxwell(Bins_mid_points, *Popt), color=Colors[i+1],marker='',ls="-")
        plt.plot(Bins_mid_points, maxwell(Bins_mid_points, 126.7737192 * 0.67), color='k',marker='',ls="-")
        
        
        #Then we perform histogram again, this time not normalised
        N_plot, Bins_plot = np.histogram(Completeness,range=(xmin,xmax),bins=nbins,density=False)
        Bins_mid_points_plot = (0.5*(Bins_plot + np.roll(Bins_plot, 1)))[1:]
        
        #Then we need to calculate expected values 
        width = Bins[1] - Bins[0] 
        Expected = np.sum(N_plot) * width * maxwell(Bins_mid_points, *Popt)
        
        #Then plot it
        plt.subplot(2,1,2)
        plt.xlim((xmin,xmax))
        plt.ylabel(r"Bin Count",size='x-large')
        plt.plot(Bins_mid_points_plot,N_plot,'x',color=Colors[i],label = "n : " + str(N_per_mle[i]))
        plt.plot(Bins_mid_points_plot, Expected, color=Colors[i+1],marker='',ls="-")
        
        
        print "Smallest number of points in any bin: ", np.min(N_plot)
        print "Chi Square: ", np.sum(  (N - maxwell(Bins_mid_points, *Popt))**2 / maxwell(Bins_mid_points, *Popt)), chisquare(N,maxwell(Bins_mid_points, *Popt),ddof=1)
        
        #print Popt, Pcov #, np.sum(  ( N_plot - Expected )**2 / Expected ) / (nbins - 3), chisquare(N_plot,Expected,ddof=3)
        print ""
        
    plt.subplots_adjust(bottom=0.12)
    legend_plot(fsize=18)

def previous_bulk_results():
    N_mle = np.array([142, 245, 128, 36])
    Z_mle = np.array([0.06,0.066,0.035,0.045])
    V_mle = np.array([260, 196, 243, 452])
    
    N_other = np.array([132, 245, 112, 112, 133, 69])
    Z_other = np.array([0.05,0.066,0.028,0.028,0.015,0.025])
    V_other = np.array([188, 249, 446, 538, 279, 541])
    
    plt.figure()
    plt.xlabel(r'$V_{bulk}$',size='x-large')
    plt.ylabel(r'$N_{SNe}$', size='x-large')
    plt.plot(V_mle, N_mle,'bo',label='ML')
    plt.plot(V_other, N_other,'ro', label = 'Other')
    legend_plot()
    
    plt.figure()
    plt.xlabel(r'$V_{bulk}$',size='x-large')
    plt.ylabel(r'$z_{SNe}$', size='x-large')
    plt.plot(V_mle, Z_mle,'bo',label='ML')
    plt.plot(V_other, Z_other,'ro', label = 'Other')
    legend_plot()
    
def supernovae_uncertainty():
    ckms = 2.99792458e5
    sigmav = lambda sigma_mu, redshift: ckms * sigma_mu * np.log(10.) * redshift * (1. + redshift/2.) / (5. * (1. + redshift))

    Redshifts = np.linspace(0,0.3, 1000)
    Sigmav_1 = np.zeros(len(Redshifts))
    Sigmav_2 = np.zeros(len(Redshifts))
    Sigmav_3 = np.zeros(len(Redshifts))

    for ii in np.arange(len(Redshifts)):
        Sigmav_1[ii] = sigmav(0.1, Redshifts[ii])
        Sigmav_2[ii] = sigmav(0.15, Redshifts[ii])
        Sigmav_3[ii] = sigmav(0.2, Redshifts[ii])

    plt.figure()
    plt.plot(Redshifts * ckms, Sigmav_1, 'b', lw=3)
    plt.plot(Redshifts * ckms, Sigmav_2, 'g', lw=3)
    plt.plot(Redshifts * ckms, Sigmav_3, 'r', lw=3)
    plt.xlabel(r'$\bar{z}$', size='xx-large')
    plt.ylabel(r'$\mathbf{\sigma_v}$ [km$\,$s$^{-1}$]', size='xx-large')
    #plt.xticks([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], size='xx-large')
    plt.yticks([0., 2000., 4000., 6000., 8000.], size='xx-large')
    plt.subplots_adjust(left=0.16, bottom=0.12)



#const_volume_poster()

const_volume()

#previous_bulk_results()

#sampling025()
#completeness025()
#cosmic025('0.062')
#cosmic025('0.25')
#cosmic025('1.0')

#cosmic025('0.125')
#cosmic025('0.5')
#cosmic025('1.0')

#cosmic_combined()
#cosmic_combined_horizontal()
#cosmic_gaussians()
#cosmic_print_values()

#sampling025_SDSS()
#completeness025_SDSS()
#sampling025_vector_SDSS()
#completeness025_vector_SDSS()

#sampling025_FoF()
#completeness025_FoF()

#samp025_FoF_mock()
#comp025_FoF_mock()

#cosmic025_maxwell()

#supernovae_uncertainty()
plt.show()