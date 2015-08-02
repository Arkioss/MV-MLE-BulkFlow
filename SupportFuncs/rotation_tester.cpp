#include <iomanip>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

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

vector<double> Radius(vector<vector<double> >XYZ)
{
 vector<double> Result(XYZ.size());
 
 for(int i=0; i<XYZ.size(); i++)
 {
  Result[i] = pow(pow(XYZ[i][0],2.0)+pow(XYZ[i][1],2.0)+pow(XYZ[i][2],2.0),0.5);
 }
 return Result;
}

double radius(vector<double> XYZ)
{
 return pow(pow(XYZ[0],2.0) + pow(XYZ[1],2.0) + pow(XYZ[2],2.0),0.5);
}

vector<double> Angle_between(vector<vector<double> > XYZ1, vector<vector<double> > XYZ2)
{
 vector<double> Result(XYZ1.size());
 double r1,r2,dot_prod;
 
 for (int i=0; i<XYZ1.size(); i++)
 {
  r1 = radius(XYZ1[i]);
  r2 = radius(XYZ2[i]);
  dot_prod = XYZ1[i][0]*XYZ2[i][0] + XYZ1[i][1]*XYZ2[i][1] + XYZ1[i][2]*XYZ2[i][2];
  Result[i] = acos(dot_prod/(r1*r2));
  cout << Result[i] * 180 / 3.1415926535 << endl;
 }
 
 return Result;
}

vector<double> Dist_between(vector<vector<double> > XYZ1, vector<vector<double> > XYZ2)
{
 vector<double> Result(XYZ1.size());
 
 for (int i=0; i<XYZ1.size(); i++)
 {
  Result[i] = pow(pow(XYZ1[i][0]-XYZ2[i][0],2.0)+pow(XYZ1[i][1]-XYZ2[i][1],2.0)+pow(XYZ1[i][2]-XYZ2[i][2],2.0),0.5);
  cout << Result[i] << endl;
 }
 return Result;
}

int main()
{

 /*
 vector<vector<double> > XYZ1, XYZ2;
 vector<vector<double> > XYZ1m,XYZ2m;
 vector<vector<double> > R;
 vector<vector<double> > Rm;
 
 XYZ1.resize(1);
 
 for (int i=0; i<1; i++)
 {
  XYZ1[i].resize(3);
 }
 
 XYZ2.resize(1);
 
 for (int i=0; i<1; i++)
 {
  XYZ2[i].resize(3);
 }
 
 XYZ1[0][0] = 100.4;
 XYZ1[0][1] = -301.0;
 XYZ1[0][2] = 321.0;
 XYZ2[0][0] = 543.0;
 XYZ2[0][1] = 123.0;
 XYZ2[0][2] = -10.1;
 
 double a = 2.2;
 double b = 1.5;
 double c = 1.3;
 
 XYZ1m = Rotate(XYZ1,a,b,c);
 XYZ2m = Rotate(XYZ2,a,b,c);
 
 Angle_between(XYZ1,XYZ2);
 Angle_between(XYZ1m,XYZ2m);
 Dist_between(XYZ1,XYZ2);
 Dist_between(XYZ1m,XYZ2m);
 */
 
 double a = 0.0;
 double b = 0.0;
 double c = 2.0 * 3.1415926535 / 4.0;
 
 vector<vector<double> > XYZv(3);
 vector<vector<double> > XYZvr(6);
 for (int i=0; i<XYZv.size(); i++)
 {
   XYZv[i].resize(6);
 }
 for (int i=0; i<XYZvr.size(); i++)
 {
   XYZvr[i].resize(6);
 }
 XYZv[0][0] = 0.0; XYZv[0][1] = 0.0; XYZv[0][2] = 0.0;
 XYZv[0][3] = 1.0; XYZv[0][4] = 1.0; XYZv[0][5] = 1.0;
 
 XYZv[1][0] = 1.0; XYZv[1][1] = 1.0; XYZv[1][2] = 1.0;
 XYZv[1][3] = 1.0; XYZv[1][4] = 1.0; XYZv[1][5] = 1.0;
 
 XYZv[2][0] = 1.0; XYZv[2][1] = 2.0; XYZv[2][2] = 3.0;
 XYZv[2][3] = 3.0; XYZv[2][4] = 2.0; XYZv[2][5] = 1.0;
 
 XYZvr = Rotate(XYZv,a,b,c);
 cout << XYZvr[0][0] << " " << XYZvr[0][1] << " "<< XYZvr[0][2] << " " << flush;
 cout << XYZvr[0][3] << " " << XYZvr[0][4] << " "<< XYZvr[0][5] << " " << endl;
 
 cout << XYZvr[1][0] << " " << XYZvr[1][1] << " "<< XYZvr[1][2] << " " << flush;
 cout << XYZvr[1][3] << " " << XYZvr[1][4] << " "<< XYZvr[1][5] << " " << endl;
 
 cout << XYZvr[2][0] << " " << XYZvr[2][1] << " "<< XYZvr[2][2] << " " << flush;
 cout << XYZvr[2][3] << " " << XYZvr[2][4] << " "<< XYZvr[2][5] << " " << endl;
 
}

