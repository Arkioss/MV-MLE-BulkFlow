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
 Author: Per Andersen, Dark Cosmology Centre
 Email: perandersen@dark-cosmology.dk
 Last revision: June 2015
 
 Compile with:
 g++-4.9 -o ACDC1 AngularCosmicDataCreate2.cpp `gsl-config --cflags --libs` -O3
 
 Use:
 GSL_RNG_SEED=10 GSL_RNG_TYPE=mt19937
 To set seed and type
 For tcsh syntax: setenv GSL_RNG_SEED "10" and setenv GSL_RNG_TYPE "mt19937"
 
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
 
 int n_want = 100000; //Number of galaxies per rotation 
 int n_rot = 80; //Number of rotations - cannot be higher than number of files/slices
 
 //Number of bins for x, y, z axis
 //Critical this is similar to that of Angular_create_histogram2.py parameters!!!
 int size_x = 15;
 int size_y = 15;
 int size_z = 15;
 
 //Set angles from Angular_create_histogram2.py here in units of Pi. Also critical!!!
 vector<double> Angle(4);
 Angle[0] = 1.0;
 Angle[1] = 0.5;
 Angle[2] = 0.25;
 Angle[3] = 0.125;
 
 //Angle[3] = 0.17;
 //Angle[1] = 0.37;
 //Angle[2] = 0.75;
 //Angle[0] = 0.062;
 
 string ROOT_DIR = "/Users/perandersen/Data/";
 string SUB_DIR = "1";
 
 // !!!PARAMETERS ARE SET HERE!!!
 
 //---------------------------------------------------------------------------------
 //-------------------------------- READING DATA -----------------------------------
 //---------------------------------------------------------------------------------
 
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
     
     cout << "Angle: " << i_angle_read << endl;
     //Reading PTF histogram data
     vector<double> Hist_density = Read_to_vector(ROOT_DIR + "BulkFlow/" + SUB_DIR+ "/Histogram/Angle_hist_" + i_angle_read + ".txt");
     vector<double> Hist_edges_x = Read_to_vector(ROOT_DIR + "BulkFlow/" + SUB_DIR+ "/Histogram/Angle_hist_" + i_angle_read + "_x.txt");
     vector<double> Hist_edges_y = Read_to_vector(ROOT_DIR + "BulkFlow/" + SUB_DIR+ "/Histogram/Angle_hist_" + i_angle_read + "_y.txt");
     vector<double> Hist_edges_z = Read_to_vector(ROOT_DIR + "BulkFlow/" + SUB_DIR+ "/Histogram/Angle_hist_" + i_angle_read + "_z.txt");
     
     //Preparing reading of HorizonRun data 
     vector<int> N_rows_hori;
     N_rows_hori = Read_to_vector_int(ROOT_DIR + "BulkFlow/DataCommon/Hori_len.txt");
     int n_cols_hori = 6;
 
     vector<vector<double> > Horizon_run;
 
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
     double slope_x = 1 / (Hist_edges_x[1]-Hist_edges_x[0]);
     double slope_y = 1 / (Hist_edges_y[1]-Hist_edges_y[0]);
     double slope_z = 1 / (Hist_edges_z[1]-Hist_edges_z[0]);
 
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
         Horizon_run = Read_to_2d_vector(ROOT_DIR + "BulkFlow/DataCommon/Hori1000_angle_" + i_read + ".dat", N_rows_hori[i_rot], n_cols_hori);
         cout << "n: " << i_rot << endl;
         
         //Get random angles and rotate Hori data accordinly
         alpha = gsl_rng_uniform(r) * 6.283185307;
         beta = gsl_rng_uniform(r) * 6.283185307;
         gamma = gsl_rng_uniform(r) * 6.283185307;
         
         Horizon_run = Rotate(Horizon_run,alpha,beta,gamma);
         
         //Cleanup and prepare for next rotation
         Indices.clear();
         Indices.resize(Horizon_run.size());
     
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
             //SANITY CHECK - THIS WILL ONLY BE 0 IF FILE WAS READ INCORRECTLY
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
       
             //Used to match index to probability from 3d PDF
             //!!!Make sure to check constant added 1000!!!
             int i_x = floor(slope_x*(Horizon_run[i_hori][0]+1000.0));
             int i_y = floor(slope_y*(Horizon_run[i_hori][1]+1000.0));
             int i_z = floor(slope_z*(Horizon_run[i_hori][2]+1000.0));
         
             int i_box = i_z + size_z*i_y + size_z*size_y * i_x;
         
             if (Hist_density[i_box] == 0.0)
             {
               i_accept -= 1;
               n_rejected += 1;
               if (i_box >= (size_x*size_y*size_z))
               {
                 //SANITY CHECK - SHOULD NOT HAVE INDEX OUTSIDE THIS RANGE
                 cout << "!!! Index outside range !!! i: " << i_box << " p: " << Hist_density[i_box] << endl;
                 exit(EXIT_FAILURE);
               }
               Indices[i_indices] = Indices[Indices.size()-1];
               Indices.pop_back();
               continue;
             }
   
             //Rolling the dice to see if the selected point is accepted or rejected
             double u = gsl_rng_uniform (r);
             if ( u < Hist_density[i_box])
             {
               //The point was accepted! We add the index to our list of accepted indices
               Accepted[i_accept] = i_hori;
               
               //In the list of candidates we copy the end to the current address, then
               //pop the back. This removes the possibility of using the same index twice.
               Indices[i_indices] = Indices[Indices.size()-1];
               Indices.pop_back();
               
               //Finally we track the number of accepted points and print the progress!
               n_accepted += 1;
               if (i_accept % 5000 == 0)
               {
                 cout << "i: " << i_accept << " " << Indices.size() << " Candidates left" << endl;
               }
             }
             else
             {
               //The point was not accepted! We reset and try again!
               i_accept -= 1;
             }
         }
         //Done finding indices for our sample - we now copy the actual data to DataOut
         for (int i=0; i<n_want; i++)
         {
             DataOut[i].resize(6);
             DataOut[i][0] = Horizon_run[Accepted[i]][0];
             DataOut[i][1] = Horizon_run[Accepted[i]][1];
             DataOut[i][2] = Horizon_run[Accepted[i]][2];
             DataOut[i][3] = Horizon_run[Accepted[i]][3];
             DataOut[i][4] = Horizon_run[Accepted[i]][4];
             DataOut[i][5] = Horizon_run[Accepted[i]][5];
         }
         //And then we save the DataOut to or data folder! Success!
         string i_save = static_cast<ostringstream*>( &(ostringstream() << i_rot) )->str();
         save_2dvector_to_file(DataOut,ROOT_DIR + "BulkFlow/" + SUB_DIR +  "/Hori_sub_cart_" + i_angle_read + "_" + i_save + ".txt");
     }
 }
 //END OF MAIN LOOP
 
 //A quick cleanup and print of the time taken to run program!
 gsl_rng_free (r);
 printf("Time taken: %.2fs\n", (double)(clock() - t_start)/CLOCKS_PER_SEC);
}