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
#include "perfuncs.h"

/*
 Compile with:
 g++ -o ACMLE3 AngularCosmicMLE2.cpp `gsl-config --cflags --libs` -O3
 
 Use:
 GSL_RNG_SEED=10 GSL_RNG_TYPE=mt19937
 To set seed and type
 For tcsh syntax: setenv GSL_RNG_SEED "10"
 
 */

vector<double> MLE_bulkflow(vector<vector<double> > Data, vector<double> V_err,double sigmastar2)
{
    double Aij, w_in, MLmag;
    double radius, rihati, rihatj;
    double vrad;
    vector<double> uML(3);
    int s;
    int ngal = int(Data.size());
    
    // Create ML_weight[3][n_SNe]
    vector<vector<double> > ML_weight(3);
    for (int i = 0; i < 3; i++) {
        ML_weight[i].resize(ngal);
    }
    
    // Create gsl matrix A_ij[3][3]
    gsl_matrix * A_ij = gsl_matrix_alloc(3, 3);
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            Aij = 0.;
            for (int n=0; n<ngal; n++)
            {
                radius = pow( pow(Data[n][0],2.0) + pow(Data[n][1],2.0) + pow(Data[n][2],2.0) ,0.5);
                rihati = Data[n][i] / radius;
                rihatj = Data[n][j] / radius;
                Aij += ((rihati*rihatj)/(V_err[n]*V_err[n] + sigmastar2));
            }
            gsl_matrix_set(A_ij,i,j,Aij);
        }
    }
    
    // Invert A_ij......................................
    gsl_matrix * A_inv = gsl_matrix_alloc(3, 3);
    gsl_permutation * perm = gsl_permutation_alloc(3);
    
    // Make LU decomposition of matrix A_ij
    gsl_linalg_LU_decomp (A_ij, perm, &s);   // This destroys original A_ij, but don't need this again
    
    // Invert the matrix A_ij
    gsl_linalg_LU_invert (A_ij, perm, A_inv);
    //..................................................
    
    // Calculate w_in, uML & MLerr
    for (int i=0; i<3; i++)
    {
        uML[i] = 0.;
        for (int n=0; n<ngal; n++)
        {
            w_in = 0.;
            for (int j=0; j<3; j++)
            {
                radius = pow( pow(Data[n][0],2.0) + pow(Data[n][1],2.0) + pow(Data[n][2],2.0) ,0.5);
                rihatj = Data[n][j] / radius;
                w_in += gsl_matrix_get(A_inv,i,j) * rihatj / (V_err[n]*V_err[n] + sigmastar2);
            }
            vrad = ( Data[n][0]*Data[n][3] + Data[n][1]*Data[n][4] + Data[n][2]*Data[n][5]) / pow( pow(Data[n][0],2.0) + pow(Data[n][1],2.0) + pow(Data[n][2],2.0) ,0.5);
            //cout << "Radial velocity: " << vrad << endl;
            uML[i] += w_in*vrad;
            ML_weight[i][n] = w_in;
        }
    }
    
    return uML;
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
 
 // !!!PARAMETERS ARE SET HERE!!!
 int n_cols = 6;
 int n_rows =45000;
 
 int n_sub = 500; //Number of galaxies used for each bulk flow calculation
 int n_draw = 1000; //Number of bulk flows calculated per rotation
 int n_sub_spher = 1689; //Number of rotations
 
 double sigma_cos2 = 250.*250.; //Cosmic variance term
 
 //Set angles from Angular_create_histogram2.py here in units of Pi. Also critical!!!
 vector<double> Angle(2);
 //Angle[0] = 1.0;
 //Angle[1] = 0.5;
 
 //Angle[0] = 0.25;
 //Angle[1] = 0.125;
 
 //Angle[0] = 0.17;
 //Angle[1] = 0.37;
 
 Angle[0] = 0.062;
 Angle[1] = 0.75;
 
 
 //string ROOT_DIR = "/Users/perandersen/Data/";
 string ROOT_DIR = "/home/per/Data/";
 string SUB_DIR = "1";
 
 // !!!PARAMETERS ARE SET HERE!!!

 //Preparing working ints and vectors
 vector<vector<double> > Hori_xyz(n_rows);
 
 int i_chosen[n_sub];
 int i_sample[n_rows];
 
 for (int i=0; i<n_rows;i++)
 {
   i_sample[i] = i;
 }
 vector<vector<double> > Hori_sub(n_sub);
 for (int i=0; i<Hori_sub.size(); i++)
 {
   Hori_sub[i].resize(6);
 }
 
 vector<double> Verr;
 vector<vector<double> >Bulk_flow(n_draw);
 for (int i=0; i<Bulk_flow.size(); i++)
 {
   Bulk_flow[i].resize(3);
 }
 
 //------------------------------- Beginning main loop ----------------------------------
 for (int i_angle=0; i_angle<Angle.size(); i_angle++)
 {
     string i_angle_read = static_cast<ostringstream*>( &(ostringstream() << Angle[i_angle]) )->str();
     cout << "Reading data..." << endl << endl;
     
     
     //These five lines are needed since c++ casts "1.0" to "1" in string. 
     int value = atoi(i_angle_read.c_str());
     if (value == 1)
     {
       i_angle_read = "1.0";
     }
     for (int i_sub=0; i_sub<n_sub_spher; i_sub++) 
     {
         cout << "n: " << i_sub << endl;
         string i_read = static_cast<ostringstream*>( &(ostringstream() << i_sub) )->str();
   
         Hori_xyz = Read_to_2d_vector(ROOT_DIR + "BulkFlow/" + SUB_DIR + "/Hori_sub_cart_" + i_angle_read + "_" + i_read + ".txt", n_rows, n_cols);
  
         for (int i_draw=0; i_draw<n_draw; i_draw++)
         {
             gsl_ran_choose(r, i_chosen, n_sub, i_sample, n_rows, sizeof(int)); 
    
             for (int i=0; i<Hori_sub.size(); i++)
             {
               Hori_sub[i][0] = Hori_xyz[i_chosen[i]][0];
               Hori_sub[i][1] = Hori_xyz[i_chosen[i]][1];
               Hori_sub[i][2] = Hori_xyz[i_chosen[i]][2];
               Hori_sub[i][3] = Hori_xyz[i_chosen[i]][3];
               Hori_sub[i][4] = Hori_xyz[i_chosen[i]][4];
               Hori_sub[i][5] = Hori_xyz[i_chosen[i]][5];
             }
             Verr = Sigma_v(Hori_sub,0.1);
             Bulk_flow[i_draw] = MLE_bulkflow(Hori_sub, Verr, sigma_cos2);
         }

     string i_save = static_cast<ostringstream*>( &(ostringstream() << i_sub) )->str();
     save_2dvector_to_file(Bulk_flow,ROOT_DIR + "BulkFlow/" + SUB_DIR + "/MLE/MLE_Bulk_flows_" + i_angle_read + "_" + i_save + "_n500.txt");
     }
 }
 printf("Time taken: %.2fs\n", (double)(clock() - t_start)/CLOCKS_PER_SEC);
}
