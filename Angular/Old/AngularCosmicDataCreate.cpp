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

/*
 Compile with:
 g++-4.9 -o ACDC AngularCosmicDataCreate.cpp `gsl-config --cflags --libs` -O3
 
 Use:
 GSL_RNG_SEED=10 GSL_RNG_TYPE=mt19937
 To set seed and type
 For tcsh syntax: setenv GSL_RNG_SEED "10"
 
 */
 
using namespace std;

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

vector<vector<double> > Read_to_2d_vector(string filename, int n_rows, int n_cols)
{/*Read single column data file to vector*/
    double read_val;
    vector<vector<double> > Result(n_rows);
    
    for (int i=0; i<n_rows; i++)
    {
        Result[i].resize(n_cols);
    }
    
    ifstream myfile(filename.c_str());
    if (!myfile)
    {
      cout << "!!!Could not read file: " << filename << endl;
      exit(-1);
    }
    
    for (int i=0; i<n_rows; i++)
    {
        for (int j=0; j<n_cols; j++)
        {
            myfile >> read_val;
            Result[i][j] = read_val;
        }
    }
    myfile.close();
    return Result;
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

int main()
{
 //Setting t0
 clock_t t_start = clock();
 
 //Setting up PRNG
 const gsl_rng_type * T;
 gsl_rng * r;
 gsl_rng_env_setup();
 T = gsl_rng_default;
 r = gsl_rng_alloc (T);
 
 int n_want = 30000; //Number of galaxies per rotation
 int n_rot = 20; //Number of rotations
 
 int size_ra = 17;
 int size_dec = 17;
 int size_mpch = 17;
 
 //---------------------------------------------------------------------------------
 //-------------------------------- READING DATA -----------------------------------
 //---------------------------------------------------------------------------------
 
 vector<double> Angle(4);
 Angle[0] = 1.0;
 Angle[1] = 0.5;
 Angle[2] = 0.25;
 Angle[3] = 0.125;
 for (int i_angle=0; i_angle<Angle.size(); i_angle++)
 {
     string i_angle_read = static_cast<ostringstream*>( &(ostringstream() << Angle[i_angle]) )->str();
     cout << "Reading data..." << endl << endl;
     
     int value = atoi(i_angle_read.c_str());
     
     if (value == 1)
     {
      i_angle_read = "1.0";
     }
     cout << "Angle: " << i_angle_read << endl;
     //Reading PTF histogram data
     vector<double> Hist_density = Read_to_vector("/Users/perandersen/Data/DataCommon/Angle_hist_" + i_angle_read + ".txt");
     vector<double> Hist_edges_ra = Read_to_vector("/Users/perandersen/Data/DataCommon/Angle_hist_" + i_angle_read + "_ra.txt");
     vector<double> Hist_edges_dec = Read_to_vector("/Users/perandersen/Data/DataCommon/Angle_hist_" + i_angle_read + "_dec.txt");
     vector<double> Hist_edges_mpch = Read_to_vector("/Users/perandersen/Data/DataCommon/Angle_hist_" + i_angle_read + "_mpch.txt");
 
     //Preparing reading of HorizonRun data 
     vector<int> N_rows_hori;
     N_rows_hori = Read_to_vector_int("/Users/perandersen/Data/DataCosmic/Hori_len.txt");
     int n_cols_hori = 6;
 
     vector<vector<double> > Horizon_run;
     vector<vector<double> > Hori_spherical;
 
     cout << "Main loop begins..." << endl << endl;
     //---------------------------------------------------------------------------------
     //------------------------------ MAIN LOOP BEGINS ---------------------------------
     //---------------------------------------------------------------------------------
 
 
     vector<int> Accepted; //Vector to hold indices of accepted galaxies
     vector<int> Indices; //Working vector to keep track of indices
     vector<vector<double> > DataOut; //To store galaxy data output
 
     int i_indices;
     int i_hori;
     int n_guess;
     int n_accepted;
     int n_rejected;
 
     //To figure out which bin it fits into
     double slope_ra = 1 / (Hist_edges_ra[1]-Hist_edges_ra[0]);
     double slope_dec = 1 / (Hist_edges_dec[1]-Hist_edges_dec[0]);
     double slope_mpch = 1 / (Hist_edges_mpch[1]-Hist_edges_mpch[0]);
 
     int alpha, beta, gamma;
 
     for (int i_rot=0; i_rot<n_rot; i_rot++)
     {
         /*Horizon_run.resize(N_rows_hori[i_rot]);
         for (int i=0; i<N_rows_hori[i]; i++)
         {
           Horizon_run[i].resize(n_cols_hori);
         }*/
         string i_read = static_cast<ostringstream*>( &(ostringstream() << i_rot) )->str();
         cout << "Reading HoriRun with " << N_rows_hori[i_rot] << " rows" << endl;
         Horizon_run = Read_to_2d_vector("/Users/perandersen/Data/DataCommon/Hori1000_angle_" + i_read + ".dat", N_rows_hori[i_rot], n_cols_hori);
         cout << "n: " << i_rot << endl;
         alpha = 0.0;//gsl_rng_uniform(r) * 6.283185307;
         beta = 0.0;//gsl_rng_uniform(r) * 6.283185307;
         gamma = 0.0;//gsl_rng_uniform(r) * 6.283185307;
         Horizon_run = Rotate(Horizon_run,alpha,beta,gamma);
         Hori_spherical = XYZ_to_RaDec(Horizon_run);
     
         Indices.clear();
         Indices.resize(Hori_spherical.size());
     
         for (int i=0; i<Indices.size(); i++)
         {
           Indices[i] = i;
         }
   
         Accepted.clear();
         Accepted.resize(n_want);
   
         for (int i=0; i<Accepted.size(); i++)
         {
           Accepted[i] = -1;
         }
   
         DataOut.clear();
         DataOut.resize(n_want);
         n_guess = 0;
         n_accepted = 0;
         n_rejected = 0;
 
         for (int i_accept=0; i_accept<n_want; i_accept++)
         {
             if (Indices.size() == 0)
             {
               cout << n_accepted << " " << n_rejected << " " << n_accepted + n_rejected << endl;
               cout << "Size = 0!!!" << endl;
             }
         
             //Select random point to test
             i_indices =  int(  gsl_rng_uniform_int(r, Indices.size())  );
             i_hori =  Indices[i_indices];
             n_guess += 1;
         
             //SANITY CHECK - THERE SHOULD BE NO DUPLICATE POINTS
             /*
             for (int i=0; i<Accepted.size(); i++)
             {
               if(Accepted[i]==i_hori)
               {
                 cout << "Duplicate!" << endl;
                 exit(EXIT_FAILURE);
               }
             }
             */
     
             int i_ra = int(floor(slope_ra*Hori_spherical[i_hori][0]));
             int i_dec = int(floor(slope_dec*(Hori_spherical[i_hori][1])));
             int i_mpch = int(floor(slope_mpch*Hori_spherical[i_hori][2]));
         
             int i_box = i_mpch + size_mpch*i_dec + size_mpch*size_dec * i_ra;
             /*
             cout << "Point: " << Hori_spherical[i_hori][2] << " " << Hori_spherical[i_hori][1] << " " << Hori_spherical[i_hori][0] << endl;
             cout << "Goes in box: " << i_box << endl;
             cout << "With P: " << Hist_density[i_box] << endl << endl;
             cout << "Index box: " << i_box << endl;
             */
         
             if (Hist_density[i_box] == 0.0)
             {
               i_accept -= 1;
               n_rejected += 1;
               if (i_box >= (size_ra*size_dec*size_mpch))
               {
                 //SANITY CHECK - SHOULD NOT HAVE INDEX OUTSIDE THIS RANGE
                 cout << "!!! i: " << i_box << " p: " << Hist_density[i_box] << endl;
                 exit(EXIT_FAILURE);
               }
               Indices[i_indices] = Indices[Indices.size()-1];
               Indices.pop_back();
               continue;
             }
   
             double u = gsl_rng_uniform (r);
             //cout << u << " " << Hist_density[i_box] << endl;
             if ( u < Hist_density[i_box])
             {
               Accepted[i_accept] = i_hori;
           
               Indices[i_indices] = Indices[Indices.size()-1];
               Indices.pop_back();
           
               n_accepted += 1;
               //cout << "Accepted!" << endl;
               if (i_accept % 100 == 0)
               {
                 cout << "i: " << i_accept << " " << Indices.size() << " Candidates left" << endl;
               }
             }
             else
             {
               i_accept -= 1;
             }
         }
         for (int i=0; i<n_want; i++)
         {
             DataOut[i].resize(6);
             DataOut[i][0] = Hori_spherical[Accepted[i]][0];
             DataOut[i][1] = Hori_spherical[Accepted[i]][1];
             DataOut[i][2] = Hori_spherical[Accepted[i]][2];
             DataOut[i][3] = Hori_spherical[Accepted[i]][3];
             DataOut[i][4] = Hori_spherical[Accepted[i]][4];
             DataOut[i][5] = Hori_spherical[Accepted[i]][5];
         }
         string i_save = static_cast<ostringstream*>( &(ostringstream() << i_rot) )->str();
         save_2dvector_to_file(DataOut,"/Users/perandersen/Data/DataCosmic/Hori_sub_spher_" + i_angle_read + "_" + i_save + ".txt");
     }
 }
 //END OF MAIN LOOP
 gsl_rng_free (r);
 printf("Time taken: %.2fs\n", (double)(clock() - t_start)/CLOCKS_PER_SEC);
}