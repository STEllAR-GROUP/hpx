///////////////////////////////////////////////////////////////////////////////
///////////////////    Gravity Dataflow Header File    ////////////////////////
//Copyright (c) 2012 Adrian Serio
//
//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
///////////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/serialization/base_object.hpp>

using namespace std;
using boost::serialization::access;

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

class Vector_container { //This class wrapps a vecotr in a shared_ptr
 private:
  boost::shared_ptr<vector<point> > p;
  
  friend class boost::serialization::access;
  template <typename Archive>
   void serialize(Archive & ar,unsigned)
   {
    ar & p;
   }
 public:
  Vector_container(): p(boost::make_shared<vector<point> >()) {};
  Vector_container(vector<point> vp)
    :p(boost::make_shared<vector<point> > (vp)) {};
  ~Vector_container(){};
  
//  Vector_container operator= (Vector_container, vector<point>) {
//   for(int i=0;i<pts.size();i++) {
//    p->at(i)
//  }
  point& operator [] (std::size_t c) {
   return p->at(c);
  }
  
  point const& operator [] (std::size_t c) const {
   return p->at(c);
  }
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
};

//Serialization
namespace boost{ namespace serialization{

 template <typename Archive>
 void serialize(Archive & ar, point & a,unsigned)
 {
   ar & a.x & a.y & a.z & a.m & a.vx & a.vy & a.fz & a.vz & a.fx & a.fy & a.fz & a.ft;
 }

 template <typename Archive>
 void serialize(Archive & ar, components & a,unsigned)
 {
   ar & a.d & a.xc & a.yc & a.zc;
 }

 template <typename Archive>
 void serialize(Archive & ar, config_f & a,unsigned)
 {
   ar & a.input;
   ar & a.output;
   ar & a.steps;
   ar & a.timestep;
 }

}}

///////////////////////////////////////////////////////////////////////////////
//Global Variables
//extern vector<vector<point> > pts_timestep;
extern bool debug;
