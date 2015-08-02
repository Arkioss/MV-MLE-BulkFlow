#ifndef __PER_FUNCS_INCLUDED__
#define __PER_FUNCS_INCLUDED__

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


using namespace std;
/*
"A central repository for c++ functions, to simplify writing, running and maintaining
code. A promise of a better tomorrow, for all per-kind.
- Per / 09-06-2015"

Functions are divided into categories, and then subdivided by alphabetical name. Func-
tions that begin with a capital letter return an array of some kind; functions that begin
with a lower case letter return single value (e.g. int, double, etc.) or are utility func-
tions.
*/


//------------------------------------ UTILITY ---------------------------------------
double linear_interp(vector<double> X, vector<double>Y, double x)
{
    double y0,y1,x0,x1; //Values needed to calculate interpolation
    int i=0, possible=0; //Value to keep track of where we are in the array, and to perform sanity check
    
    
    //This loop is run to find the appropriate values of x and y to use for interpolation
    while (X[i] <= x)
    {
        //cout << i << " " << exp(X[i]) << " " << exp(x) << endl;
        i += 1;
        possible = 1;
    }
    
    if (i>0)
    {
        if (pow(X[i-1]-x,2.0) < pow(X[i]-x,2.0))
        {
            i -= 1;
        }
        else if (pow(X[i+1]-x,2.0) < pow(X[i]-x,2.0))
        {
            i += 1;
        }
    }
    x0 = X[i];
    x1 = X[i+1];
    y0 = Y[i];
    y1 = Y[i+1];
    
    if(possible == 0)
    {
        cout << "Impossible to Interpolate! " << endl;
        exit(-1);
    }
    return y0 + (y1-y0) * ( (x-x0) / (x1-x0) );
}

//---------------------------------- INPUT/OUTPUT --------------------------------------
vector<vector<double> > Read_to_2d_vector(string filename, int n_rows, int n_cols)
{/*Read two column data file to vector of vectors*/
    
    
    double read_val;
    vector<vector<double> > Result(n_rows);
    
    //Resizing output vector
    for (int i=0; i<n_rows; i++)
    {
        Result[i].resize(n_cols);
    }
    
    //Checking that file exists and can be read, if able then file is open
    ifstream myfile(filename.c_str());
    if (!myfile)
    {
      cout << "!!!Could not read file: " << filename << endl;
      exit(-1);
    }
    
    //Pushing data into vectors
    for (int i=0; i<n_rows; i++)
    {
        for (int j=0; j<n_cols; j++)
        {
            myfile >> read_val;
            Result[i][j] = read_val;
        }
    }
    
    //Closing file
    myfile.close();
    
    return Result;
}

vector<double> Read_to_vector(string filename)
{/*Read single column data file to vector*/
    double read_val;
    vector<double> Result;
    
    
    ifstream myfile(filename.c_str());
    if (!myfile)
    {
      cout << "!!!Could not read file: " << filename << endl;
      exit(-1);
    }
    while (myfile.good())
    {
        myfile >> read_val;
        Result.push_back(read_val);
    }
    myfile.close();
    return Result;
}

vector<int> Read_to_vector_int(string filename)
{/*Read single column data file to vector*/
    int read_val;
    vector<int> Result;
    ifstream myfile(filename.c_str());
    if (!myfile)
    {
      cout << "!!!Could not read file: " << filename << endl;
      exit(-1);
    }
    
    while (myfile.good())
    {
        myfile >> read_val;
        Result.push_back(read_val);
    }
    myfile.close();
    return Result;
}

void save_vector_to_file(vector<double> q, string file_name)
{
    ofstream file;
    file.open(file_name.c_str());
    for (int i=0; i<q.size(); ++i)
        file << q[i] << "\n";
    file.close();
}

void save_2dvector_to_file(vector<vector<double> > Q, string file_name)
{
    ofstream file;
    file.open(file_name.c_str());
    for (int i=0; i<Q.size(); ++i)
    {
        for (int j=0; j<Q[i].size(); j++)
        {
            file << Q[i][j] << " ";
        }
        file << "\n";
    }
    file.close();
}
//----------------------------- COORDINATE TRANSFORMATIONS -----------------------------
vector<vector<double> > RaDec_to_xyz(vector<vector<double> > RaDec)
{
 
 if(RaDec[0].size()==3)
 {
    
    vector<vector<double> > XYZ(RaDec.size());
    
    for (int i=0; i<XYZ.size(); i++)
    {
        XYZ[i].resize(3);
    }
    
    double theta, phi, radius; //To ease conversion to xyz
    
    for (int i=0; i<XYZ.size(); i++)
    {
        phi = RaDec[i][0] * 3.1415926535 / 180.0;
        theta = (90.0 - RaDec[i][1]) * 3.1415926535 / 180.0;
        radius = RaDec[i][2];
        
        XYZ[i][0] = radius * sin(theta) * cos(phi);
        XYZ[i][1] = radius * sin(theta) * sin(phi);
        XYZ[i][2] = radius * cos(theta);
        //cout << XYZ[i][0] << " " << XYZ[i][1] << " " << XYZ[i][2] << endl;
    }
    return XYZ;
 }
 if(RaDec[0].size()==6)
 {
    
    vector<vector<double> > XYZ(RaDec.size());
    
    for (int i=0; i<XYZ.size(); i++)
    {
        XYZ[i].resize(6);
    }
    
    double theta, phi, radius; //To ease conversion to xyz
    
    for (int i=0; i<XYZ.size(); i++)
    {
        phi = RaDec[i][0] * 3.1415926535 / 180.0;
        theta = (90.0 - RaDec[i][1]) * 3.1415926535 / 180.0;
        radius = RaDec[i][2];
        
        XYZ[i][0] = radius * sin(theta) * cos(phi);
        XYZ[i][1] = radius * sin(theta) * sin(phi);
        XYZ[i][2] = radius * cos(theta);
        XYZ[i][3] = RaDec[i][3];
        XYZ[i][4] = RaDec[i][4];
        XYZ[i][5] = RaDec[i][5];
        //cout << XYZ[i][0] << " " << XYZ[i][1] << " " << XYZ[i][2] << endl;
    }
    return XYZ;
 }
}

vector<vector<double> > XYZ_to_RaDec(vector<vector<double> > XYZ)
{
 if (XYZ[0].size() == 3)
 {
    vector<vector<double> > RaDec(XYZ.size());
    
    for (int i=0; i<RaDec.size(); i++)
    {
        RaDec[i].resize(3);
    }
    
    double theta, phi, radius; //To ease conversion to xyz
    
    for (int i=0; i<XYZ.size(); i++)
    {
        radius = pow( pow(XYZ[i][0],2.0) + pow(XYZ[i][1],2.0) + pow(XYZ[i][2],2.0) ,0.5);
        theta = acos(XYZ[i][2]/radius);
        phi = atan2(XYZ[i][1],XYZ[i][0]);
        
        RaDec[i][0] = phi * 180.0 / 3.1415926535;
        if (RaDec[i][0] < 0.0)
        {
            RaDec[i][0] += 360.0;
        }
        RaDec[i][1] = 90.0 - theta * 180.0 / 3.1415926535;
        RaDec[i][2] = radius;
    }
    return RaDec;
 }
 if (XYZ[0].size() == 6)
 {
    vector<vector<double> > RaDec(XYZ.size());
    
    for (int i=0; i<RaDec.size(); i++)
    {
        RaDec[i].resize(6);
    }
    
    double theta, phi, radius; //To ease conversion to xyz
    
    for (int i=0; i<XYZ.size(); i++)
    {
        radius = pow( pow(XYZ[i][0],2.0) + pow(XYZ[i][1],2.0) + pow(XYZ[i][2],2.0) ,0.5);
        theta = acos(XYZ[i][2]/radius);
        phi = atan2(XYZ[i][1],XYZ[i][0]);
        
        RaDec[i][0] = phi * 180.0 / 3.1415926535;
        if (RaDec[i][0] < 0.0)
        {
            RaDec[i][0] += 360.0;
        }
        RaDec[i][1] = 90.0 - theta * 180.0 / 3.1415926535;
        RaDec[i][2] = radius;
        RaDec[i][3] = XYZ[i][3];
        RaDec[i][4] = XYZ[i][4];
        RaDec[i][5] = XYZ[i][5];
    }
    return RaDec;
 }
}

vector<vector<double> > Rotate(vector<vector<double> > XYZ, double a, double b, double c)
{
 vector<vector<double> > Result(XYZ.size());
 
 if (XYZ[0].size() == 3)
 {
   for (int i=0; i<Result.size(); i++)
   {
    Result[i].resize(3);
   }
   
   for (int i=0; i<XYZ.size(); i++)
   {
    
    Result[i][0] = cos(b)*cos(c)*XYZ[i][0] - cos(b)*sin(c)*XYZ[i][1] + sin(b)*XYZ[i][2];
    
    Result[i][1] = sin(a)*sin(b)*cos(c)*XYZ[i][0] + cos(a)*sin(c)*XYZ[i][0]
     - sin(a)*sin(b)*sin(c)*XYZ[i][1] + cos(a)*cos(c)*XYZ[i][1] - sin(a)*cos(b)*XYZ[i][2];
    
    Result[i][2] = -cos(a)*sin(b)*cos(c)*XYZ[i][0] + sin(a)*sin(c)*XYZ[i][0]
     + cos(a)*sin(b)*sin(c)*XYZ[i][1] + sin(a)*cos(c)*XYZ[i][1] + cos(a)*cos(b)*XYZ[i][2];
   }
 return Result;
 }
 
 else if (XYZ[0].size() == 6)
 { 
   for (int i=0; i<Result.size(); i++)
   {
    Result[i].resize(6);
   }
   
   for (int i=0; i<XYZ.size(); i++)
   {
    
    Result[i][0] = cos(b)*cos(c)*XYZ[i][0] - cos(b)*sin(c)*XYZ[i][1] + sin(b)*XYZ[i][2];
    
    Result[i][1] = sin(a)*sin(b)*cos(c)*XYZ[i][0] + cos(a)*sin(c)*XYZ[i][0]
     - sin(a)*sin(b)*sin(c)*XYZ[i][1] + cos(a)*cos(c)*XYZ[i][1] - sin(a)*cos(b)*XYZ[i][2];
    
    Result[i][2] = -cos(a)*sin(b)*cos(c)*XYZ[i][0] + sin(a)*sin(c)*XYZ[i][0]
     + cos(a)*sin(b)*sin(c)*XYZ[i][1] + sin(a)*cos(c)*XYZ[i][1] + cos(a)*cos(b)*XYZ[i][2];
     
    Result[i][3] = cos(b)*cos(c)*XYZ[i][3] - cos(b)*sin(c)*XYZ[i][4] + sin(b)*XYZ[i][5];
    
    Result[i][4] = sin(a)*sin(b)*cos(c)*XYZ[i][3] + cos(a)*sin(c)*XYZ[i][3]
     - sin(a)*sin(b)*sin(c)*XYZ[i][4] + cos(a)*cos(c)*XYZ[i][4] - sin(a)*cos(b)*XYZ[i][5];
    
    Result[i][5] = -cos(a)*sin(b)*cos(c)*XYZ[i][3] + sin(a)*sin(c)*XYZ[i][3]
     + cos(a)*sin(b)*sin(c)*XYZ[i][4] + sin(a)*cos(c)*XYZ[i][4] + cos(a)*cos(b)*XYZ[i][5];
   }
 return Result;
 }
 
 else 
 {
   cout << "DANGER!" << endl;
   return Result;
 }
 
}

//------------------------------------ COSMOLOGY ---------------------------------------
 double e_inv(double z)
{/*Returns the inverse of E(z) for flat matter lambda cosmology*/
    double const omega_m = 0.3;
    return pow(omega_m*pow(1+z,3.0)+(1-omega_m),-0.5);
}

double lum_distance_per_h(double z)
{/*Returns the luminosity distance of redshift z for E(z) function*/
    double int_sum=0.0,int_tracker=0.0;
    double delta_z = z / 100000.0;
    while (int_tracker + delta_z < z)
    {
        int_sum += delta_z * (e_inv(int_tracker)+e_inv(int_tracker+delta_z)) / 2.0;
        int_tracker += delta_z;
    }
    return (1+z)*int_sum * 2.99792e5 / 100.0;
}

vector<double> Dist_per_h_to_redshift(vector<double> Distance)
{
    //a) Create interpolation table
    //First check if it has already been created
    vector<double> Redshift_interp, Lum_dist_interp;
    vector<double> Redshift_r(Distance.size());
    double z_interp_max = 2.0;
    double dz = 0.0005;	
    double z_tracker = 0.0;
    
    //Checking that files exist
    ifstream myfile1("/Users/perandersen/Data/BulkFlow/DataCommon/interp_redshift_h.txt");
    ifstream myfile2("/Users/perandersen/Data/BulkFlow/DataCommon/interp_lum_dist_h.txt");
    if ( (!myfile1) or (!myfile2))
    {
        cout << "File doesn't exist. Calculating and saving new interpolation table..." << endl;
        Redshift_interp.clear();
        Lum_dist_interp.clear();
        while (z_tracker + dz < z_interp_max)
        {
            Redshift_interp.push_back(z_tracker);
            Lum_dist_interp.push_back(lum_distance_per_h(z_tracker));
            z_tracker += dz;
        }
        save_vector_to_file(Redshift_interp,"interp_redshift_h.txt");
        save_vector_to_file(Lum_dist_interp,"interp_lum_dist_h.txt");
        cout << "Done!" << endl << endl;
    }
    else
    {
    Redshift_interp = Read_to_vector("/Users/perandersen/Data/BulkFlow/DataCommon/interp_redshift_h.txt");
    Lum_dist_interp = Read_to_vector("/Users/perandersen/Data/BulkFlow/DataCommon/interp_lum_dist_h.txt");
    
    if ( abs(Redshift_interp.size() - (z_interp_max/dz)) > 2)
    {
        cout << "File is wrong. Calculating and saving new interpolation table..." << endl;
        Redshift_interp.clear();
        Lum_dist_interp.clear();
        while (z_tracker + dz < z_interp_max)
        {
            Redshift_interp.push_back(z_tracker);
            Lum_dist_interp.push_back(lum_distance_per_h(z_tracker));
            z_tracker += dz;
        }
        save_vector_to_file(Redshift_interp,"interp_redshift_h.txt");
        save_vector_to_file(Lum_dist_interp,"interp_lum_dist_h.txt");
        cout << "Done!" << endl << endl;
    }
    
    }
    //b) Then calculating the actual redshifts
    for (int i=0; i<Distance.size(); i++)
    {
        if (Distance[i] < 0.0)
        {
            Redshift_r[i] = -linear_interp(Lum_dist_interp,Redshift_interp,-Distance[i]);
        }
        else
        {
            Redshift_r[i] = linear_interp(Lum_dist_interp,Redshift_interp,Distance[i]);
        }
    }
    return Redshift_r;
}

vector<double> Sigma_v(vector<vector<double> > Data, double sigma_mu)
{
 vector<double> Result(Data.size());
 vector<double> Radius(Data.size());
 vector<double> Redshift_r;
 
 for (int i=0; i<Result.size(); i++)
 {
  Radius[i] = pow( Data[i][0]*Data[i][0] + Data[i][1]*Data[i][1] + Data[i][2]*Data[i][2] ,0.5);
 }
 
 Redshift_r = Dist_per_h_to_redshift(Radius);

 for (int i=0; i<Result.size(); i++)
 {
  Result[i] = 1.380595234e5 * sigma_mu * Redshift_r[i] *(1.0 + Redshift_r[i]/2.0) / (1.0 + Redshift_r[i]);
 }

 return Result;
}

#endif

