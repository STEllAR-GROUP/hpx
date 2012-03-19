///////////////////////////////////////////////////////////////////////////////
///////////////////    Gravity Header File    /////////////////////////////////
//Copyright (c) 2012 Adrian Serio
//
//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
///////////////////////////////////////////////////////////////////////////////

#include <string>

using namespace std;
class point {
 public:
  point(double xx, double yy, double zz, double mm, double vxx, double vyy,
   double vzz)
   :x(xx),y(yy),z(zz),m(mm),vx(vxx),vy(vyy),vz(vzz){}
  point():x(0),y(0),m(0),vx(0),vy(0),vz(0),fx(0),fy(0),fz(0),ft(0) {}
  ~point(){}
  double x;  //x-coordinate
  double y;  //y-coordinate
  double z;  //z-coordinate
  double m;  //mass
  double vx; //velocity in the x direction
  double vy; //velocity in the y direction
  double vz; //velocity in the z direction
  double fx; //force in the x direction
  double fy; //force in the y direction
  double fz; //force in the z direction
  double ft; //the sum of the forces
};  

struct components {
  double d;  //distance
  double xc; //x-component
  double yc; //y-component
  double zc; //z-component
};

struct config_f {
 string input;
 string output;
 uint64_t steps;
 uint64_t timestep;
 uint64_t print;
 uint64_t num_cores;

 template <typename Archive>
 void serialize(Archive & ar, unsigned)
 {
   ar & input;
   ar & output;
   ar & steps;
   ar & timestep;
 }
};

///////////////////////////////////////////////////////////////////////////////
//Global Variables
extern vector<vector<point> > pts_timestep;
extern bool debug;
