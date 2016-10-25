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
g++-5 -o CMLE2 CosmicMLE3.cpp `gsl-config --cflags --libs` -O3

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
 int n_rows = 45000;
 
 int n_sub = 8000; //Number of galaxies used for each bulk flow calculation

 int n_draw = 5; //Number of bulk flows calculated
 int n_rot = 1689; //Number of rotations
 
 double sigma_cos2 = 250.0*250.0; //Cosmic variance term
 
 
 string ROOT_DIR = "/home/per/Data/"; 
 //string ROOT_DIR = "/Users/perandersen/Data/";
 string SUB_DIR = "8A";
 
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
vector<vector<double> > Bulk_flow(n_draw);
for (int i=0; i<Bulk_flow.size(); i++)
{
  Bulk_flow[i].resize(3);
}
 

 for (int i_rot=0; i_rot<n_rot; i_rot++)  
 {
     string i_rot_str = static_cast<ostringstream*>( &(ostringstream() << i_rot) )->str();
     Hori_xyz = Read_to_2d_vector(ROOT_DIR + "BulkFlow/1/Hori_sub_cart_1.0_" + i_rot_str + ".txt", n_rows, n_cols);
     cout << "n: " << i_rot << endl;
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
     
     string i_n_sub = static_cast<ostringstream*>( &(ostringstream() << n_sub) )->str();
     save_2dvector_to_file(Bulk_flow,ROOT_DIR + "BulkFlow/" + SUB_DIR + "/MLE/MLE_Bulk_flows_1.0_" + i_rot_str + "_" + i_n_sub + ".txt");
 }
 printf("Time taken: %.2fs\n", (double)(clock() - t_start)/CLOCKS_PER_SEC);
}


