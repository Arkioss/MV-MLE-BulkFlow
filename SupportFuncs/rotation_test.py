from __future__ import division
import numpy as np


x,y,z = 2,0,0
a,b,c = np.pi/2,np.pi,np.pi/5

def rotate(x,y,z,a,b,c):
 xm = np.cos(b)*np.cos(c)*x - np.cos(b)*np.sin(c)*y + np.sin(b)*z
 ym = np.sin(a)*np.sin(b)*np.cos(c)*x + np.cos(a)*np.sin(c)*x - np.sin(a)*np.sin(b)*np.sin(c)*y + np.cos(a)*np.cos(c)*y - np.sin(a)*np.cos(b)*z
 zm = -np.cos(a)*np.sin(b)*np.cos(c)*x + np.sin(a)*np.sin(c)*x + np.cos(a)*np.sin(b)*np.sin(c)*y + np.sin(a)*np.cos(c)*y + np.cos(a)*np.cos(b)*z
 return None

def radius(x,y,z):
 return np.sqrt(x**2 + y**2 +z**2)

def dist_between(x1,y1,z1,x2,y2,z2):
 return np.sqrt( (x1-x2)**2 + (y1-y2)**2 +(z1-z2)**2 )

def angle_between(x1,y1,z1,x2,y2,z2) :
 if (x1==x2):
  if (y1==y2):
   if (z1==z2):
    return 0
 print radius(x1,y1,z1), radius(x2,y2,z2), x1*x2+y1*y2+z1*z2
 return np.arccos( (x1*x2+y1*y2+z1*z2) / (radius(x1,y1,z1)*radius(x2,y2,z2)) )

print angle_between(0,0,1,0,0,-1) * 180 / np.pi