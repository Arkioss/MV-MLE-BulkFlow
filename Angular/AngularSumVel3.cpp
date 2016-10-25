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
g++ -o ASV1 AngularSumVel3.cpp `gsl-config --cflags --libs` -O3
 
Use:
GSL_RNG_SEED=10 GSL_RNG_TYPE=mt19937
To set seed and type
For tcsh syntax: setenv GSL_RNG_SEED "10"
 */

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

 int n_sub_spher = 1689; //Number of rotations
 
 //Set angles from Angular_create_histogram2.py here in units of Pi. Also critical!!!
 vector<double> Angle(4);
 
 /*
 Angle[0] = 1.0;
 Angle[1] = 0.75;
 Angle[2] = 0.5;
 Angle[3] = 0.37;
 */
 
 Angle[0] = 0.25;
 Angle[1] = 0.17;
 Angle[2] = 0.125;
 Angle[3] = 0.062;
 
 
 vector<double> Radius_eff(8);

 Radius_eff[0] = 50.;
 Radius_eff[1] = 50.;
 Radius_eff[2] = 50.;
 Radius_eff[3] = 50.;
 Radius_eff[4] = 50.;
 Radius_eff[5] = 50.;
 Radius_eff[6] = 50.;
 Radius_eff[7] = 50.;

 
 /*
 Radius_eff[0] = 100;
 Radius_eff[1] = 150;
 Radius_eff[2] = 200;
 Radius_eff[3] = 250;
 Radius_eff[4] = 300;
 Radius_eff[5] = 350;
 Radius_eff[6] = 400;
 */
 
 /*
 Radius_eff[0] = 250;
 Radius_eff[1] = 280;
 Radius_eff[2] = 330;
 Radius_eff[3] = 400;
 Radius_eff[4] = 500;
 Radius_eff[5] = 600;
 Radius_eff[6] = 800;
 */
 double r_eff; //Only used for gaussian weights
 
 //string ROOT_DIR = "/Users/perandersen/Data/";
 string ROOT_DIR = "/home/per/Data/";
 string SUB_DIR = "1";
 
 // !!!PARAMETERS ARE SET HERE!!!

 //Preparing working ints and vectors
 vector<vector<double> > Hori_xyz(n_rows);
 
 double sum_vx = 0.0;
 double sum_vy = 0.0;
 double sum_vz = 0.0;
 
 double radius;
 double vector_factor = 1.;
 double weight;
 double weight_sum = 0.;
 
 vector<vector<double> >Bulk_flow(n_sub_spher);
 for (int i=0; i<Bulk_flow.size(); i++)
 {
   Bulk_flow[i].resize(3);
 }
 
 vector<double> bulk_flow_holder(3);
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
       
         Hori_xyz = Read_to_2d_vector(ROOT_DIR + "BulkFlow/" + SUB_DIR + "/Hori_sub_cart_" + i_angle_read + "_" + i_read + ".txt", n_rows, n_cols);
     
         for (int i_xyz=0; i_xyz<Hori_xyz.size(); i_xyz++)
         {
           //Gaussian
           //radius = pow(pow(Hori_xyz[i_xyz][0],2.0)+pow(Hori_xyz[i_xyz][1],2.0)+pow(Hori_xyz[i_xyz][2],2.0),0.5);
           //r_eff = Radius_eff[i_angle];
           //weight = radius * radius * exp(  -(radius*radius) / (2.0 * r_eff*r_eff)  );
           
           //Tophat
           weight = 1.0;
           
           weight_sum += weight;
           
           //If we want to use only line-of-sight components
           //vector_factor = (Hori_xyz[i_xyz][0]*Hori_xyz[i_xyz][3] + Hori_xyz[i_xyz][1]*Hori_xyz[i_xyz][4] + Hori_xyz[i_xyz][2]*Hori_xyz[i_xyz][5])
           //             / (Hori_xyz[i_xyz][0]*Hori_xyz[i_xyz][0] + Hori_xyz[i_xyz][1]*Hori_xyz[i_xyz][1] + Hori_xyz[i_xyz][2]*Hori_xyz[i_xyz][2]);
                         
           sum_vx += vector_factor * Hori_xyz[i_xyz][3] * weight;
           sum_vy += vector_factor * Hori_xyz[i_xyz][4] * weight;
           sum_vz += vector_factor * Hori_xyz[i_xyz][5] * weight;
           
         }
         sum_vx = sum_vx / weight_sum;
         sum_vy = sum_vy / weight_sum;
         sum_vz = sum_vz / weight_sum;
         
         bulk_flow_holder[0] = sum_vx;
         bulk_flow_holder[1] = sum_vy;
         bulk_flow_holder[2] = sum_vz;
         
         Bulk_flow[i_sub] = bulk_flow_holder;
         sum_vx = 0.0;
         sum_vy = 0.0;
         sum_vz = 0.0;
         weight_sum = 0.0;
     }
     save_2dvector_to_file(Bulk_flow,ROOT_DIR + "BulkFlow/" + SUB_DIR + "/Sum/Bulk_flows_sum_" + i_angle_read + ".txt");
 }
 printf("Time taken: %.2fs\n", (double)(clock() - t_start)/CLOCKS_PER_SEC);
}
