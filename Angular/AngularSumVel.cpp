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
 g++-4.9 -o ASV AngularSumVel.cpp `gsl-config --cflags --libs` -O3
 
 Use:
 GSL_RNG_SEED=10 GSL_RNG_TYPE=mt19937
 To set seed and type
 For tcsh syntax: setenv GSL_RNG_SEED "10"
 
 */
 
using namespace std;

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
 int n_rows = 10000;

 int n_sub_spher = 50; //Number of rotations
 
 //Set angles from Angular_create_histogram2.py here in units of Pi. Also critical!!!
 vector<double> Angle(7);
 Angle[0] = 1.0;
 Angle[1] = 0.75;
 Angle[2] = 0.5;
 Angle[3] = 0.37;
 Angle[4] = 0.25;
 Angle[5] = 0.17;
 Angle[6] = 0.125;
 
 vector<double> Radius_eff(7);
 Radius_eff[0] = 100;
 Radius_eff[1] = 150;
 Radius_eff[2] = 200;
 Radius_eff[3] = 250;
 Radius_eff[4] = 300;
 Radius_eff[5] = 350;
 Radius_eff[6] = 400;
 
 double r_eff = 80.0; //Only used for gaussian weights
 
 string ROOT_DIR = "/Users/perandersen/Data/";
 string SUB_DIR = "2";
 
 // !!!PARAMETERS ARE SET HERE!!!

 //Preparing working ints and vectors
 vector<vector<double> > Hori_xyz(n_rows);
 
 vector<double> Bulk_flow(n_sub_spher);
 
 double sum_vx = 0.0;
 double sum_vy = 0.0;
 double sum_vz = 0.0;
 
 double radius;
 double weight;
 double weight_sum = 0;
 
 //Beginning main loop
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
       
         Hori_xyz = Read_to_2d_vector(ROOT_DIR + "BulkFlow/" + SUB_DIR + "/MV/Hori_sub_cart_" + i_angle_read + "_" + i_read + ".txt", n_rows, n_cols);
     
         
         for (int i_xyz=0; i_xyz<Hori_xyz.size(); i_xyz++)
         {
           //Gaussian weights
           radius = pow(pow(Hori_xyz[i_xyz][0],2.0)+pow(Hori_xyz[i_xyz][1],2.0)+pow(Hori_xyz[i_xyz][2],2.0),0.5);
           
           //Gaussian
           weight = radius * radius * exp(  -(radius*radius) / (2.0 * r_eff*r_eff)  );
           //Tophat
           //weight = 1.0;
           
           weight_sum += weight;
           
           sum_vx += Hori_xyz[i_xyz][3] * weight;
           sum_vy += Hori_xyz[i_xyz][4] * weight;
           sum_vz += Hori_xyz[i_xyz][5] * weight;
           
         }
         //Gaussian weights
         sum_vx = sum_vx / weight_sum;
         sum_vy = sum_vy / weight_sum;
         sum_vz = sum_vz / weight_sum;
         
         Bulk_flow[i_sub] = pow(pow(sum_vx,2.0)+pow(sum_vy,2.0)+pow(sum_vz,2.0),0.5);
         sum_vx = 0.0;
         sum_vy = 0.0;
         sum_vz = 0.0;
         weight_sum = 0.0;
     }
     save_vector_to_file(Bulk_flow,ROOT_DIR + "BulkFlow/" + SUB_DIR + "/Sum/Bulk_flows_sum_" + i_angle_read + ".txt");
 }
 printf("Time taken: %.2fs\n", (double)(clock() - t_start)/CLOCKS_PER_SEC);
}