///////////////////////////////////////////////////////////////////////////////
///////////////////////////////     GRAVITY    ////////////////////////////////
//This program calculates the effect of gravity on a set of points in three 
//dimentions. It uses a "coordinate file" to define the points and gives the
//user ability to adjust the number of iterations and time that passes for each 
//iteration. These files and values are changed in the configuraion file,
//in this case "gravconfig.conf".
//
//Problem:
//
//Remember to change the program version when program is updated! 
///////////////////////////////////////////////////////////////////////////////
//Copyright (c) 2011 Adrian Serio
//
//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
///////////////////////////////////////////////////////////////////////////////
//Units are in m, kg, s, and therefore N.
///////////////////////////////////////////////////////////////////////////////
#include<iostream>
#include<fstream>
#include<sstream>
#include<cmath>
#include<cstdlib>
#include<string>
#include<list>
#include<vector>
#include<algorithm>
#include<stdexcept>

#include<boost/cstdint.hpp>
#include<boost/format.hpp>

#include<hpx/hpx.hpp>
#include<hpx/hpx_fwd.hpp>
#include<hpx/hpx_init.hpp>
#include<hpx/runtime/actions/plain_action.hpp>
#include<hpx/runtime/components/plain_component_factory.hpp>
#include<hpx/util/high_resolution_timer.hpp>
#include<hpx/lcos/eager_future.hpp>

#include "gravity.hpp"

using namespace std;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::actions::plain_result_action1;
using hpx::actions::plain_result_action5;
using hpx::actions::plain_action3;
using hpx::actions::plain_action4;
using hpx::lcos::eager_future;
using hpx::lcos::promise;
using hpx::lcos::wait;
using hpx::util::high_resolution_timer;
using hpx::init;
using hpx::finalize;
using hpx::find_here;
using hpx::get_os_thread_count;

// dummy serialize function in order to be able to pass a promise, or
// vector of promises to a future
// WARNING: this will not work in distributed!
namespace boost
{
    template <typename Archive>
    void serialize(Archive & ar, hpx::lcos::promise<void> &, unsigned)
    {
    }
}

///////////////////////////////////////////////////////////////////////////////
#if 0
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
  double d;  //distance
  double xc; //x-component
  double yc; //y-component
  double zc; //z-component
};
struct config_f {
 string input;
 string output;
 int steps;
 int timestep;
 int num_cores;

 template <typename Archive>
 void serialize(Archive & ar, unsigned)
 {
   ar & input;
   ar & output;
   ar & steps;
   ar & timestep;
 }
};
#endif
///////////////////////////////////////////////////////////////////////////////
//Forward Declarations
vector<promise<void> > calc(int k,int t);
vector<point> dist(int t,int i,int l);
void move(vector<promise<void> > const& cfp,config_f const& param,int k,int t);
void printval(promise<void> const & mp,int k,int t,ofstream &coorfile,
               ofstream &trbst);
void closefile(ofstream &coorfile,ofstream &trbst,ofstream &notes,
               config_f& param,float ct);
void loadconfig(config_f& param,variables_map& vm);
void setup(ofstream &coorfile,ofstream &trbst,int k);
//vector<point> createvecs(ifstream &ist);
vector<point> createvecs(config_f& param);
void calc_force(int k,int t,int i);

///////////////////////////////////////////////////////////////////////////////
//Action Declarations
typedef plain_action3<
     int,         // argument 
     int,         // argument
     int,         // argument
     calc_force   // function
 > calc_force_action;
HPX_REGISTER_PLAIN_ACTION(calc_force_action);

typedef plain_action4< 
     vector<promise<void> > const &,
     config_f const &,
     int,
     int,
     move
 > move_action;
HPX_REGISTER_PLAIN_ACTION(move_action);

///////////////////////////////////////////////////////////////////////////////
//Future Declarations
typedef eager_future<calc_force_action> calc_force_future;
typedef eager_future<move_action> move_future;

///////////////////////////////////////////////////////////////////////////////
//Global Variables
vector<vector<point> > pts_timestep;
bool debug;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm) 
{
  {
     config_f param;
     string j; // Buffer
     int k=0; //Number of points in simulation
     float ct=0; //Computation time
     vector<promise<void> > cfp; // calc_force promises
     promise<void> mp; //move promise

     debug=vm.count("debug");
     cout <<"Config File Location: "<< vm["config-file"].as<std::string>() 
          << "\n";
     param.num_cores=get_os_thread_count();

    //This section opens an input file
     loadconfig(param,vm); //load the configuraton file
     ifstream ist(param.input.c_str()); //Name of input file ="param.input"
     if(!ist) {
      cout<<"Cannot open input file!\n";
     }

    // This section creates and opens the coordinate output file
     ofstream coorfile;
     j=param.output+".cf.txt";
     coorfile.open(j.c_str());

    //This section opens the trb output file
     j=param.output+".trb.txt";
     ofstream trbst;
     trbst.open(j.c_str());

    //This section opens and creates a documentation output file
     j=param.output+".notes.txt";
     ofstream notes;
     notes.open(j.c_str());

    //This section sets up the rest of the program
//     pts_timestep = vector<vector<point> >(param.steps+1,createvecs(ist));
     pts_timestep = vector<vector<point> >(param.steps+1,createvecs(param));
     k=pts_timestep[0].size();
     setup(coorfile,trbst,k); // sets up the output files
     
     high_resolution_timer ht;
    //This section outlines the program
     for (int t=0;t<param.steps;t++) {
      if (debug) cout<<"\nFor step "<<t<<":\n";
      cfp=calc(k,t);  //This is the calculation of force
      mp=move_future(find_here(),cfp,param,k,t);
      printval(mp,k,t,coorfile,trbst); //Writes output
     }
     ct=ht.elapsed();
     cout<<"Computation Time: "<<ct<<" [s]\n";
     closefile(coorfile,trbst,notes,param,ct);
 }
 hpx::finalize();
 return 0;
}

int main(int argc, char*argv[])
{
 //Configure application-specific options
 options_description
  desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

 desc_commandline.add_options()
  ( "config-file",value<std::string>(),
   "path to the configuration file")
  ("debug","prints out coordinates for each step");
 return init(desc_commandline, argc, argv);
}

vector<promise<void> > calc(int k,int t) {
  
 vector<promise<void> > cfp;
 for (int i=0;i<k;++i) { //moves through the points ie 1, 2, etc.
  cfp.push_back(calc_force_future(find_here(),k,t,i));
 }
 return cfp;
}

vector<point> dist(int t,int i,int l) {
 double rx; //distance in the x direction
 double ry; //distance in the y direction
 double rz; //distance in the z direction
 double r2t; //distance between the two points squared
 double rt; //distance between the two points

 rx=abs(pts_timestep[t][l].x-pts_timestep[t][i].x);
 ry=abs(pts_timestep[t][l].y-pts_timestep[t][i].y);
 rz=abs(pts_timestep[t][l].z-pts_timestep[t][i].z);
 r2t=(rx*rx)+(ry*ry)+(rz*rz);
 rt=sqrt(r2t);

 vector<point>comp(1);
 
 comp[0].d=r2t;
 comp[0].xc=rx/rt;
 comp[0].yc=ry/rt;
 comp[0].zc=rz/rt;

 return comp;
}

void move(vector<promise<void> >const& cfp,config_f const & param,int k,int t) {
 int tn=t+1;
 double timestep=param.timestep; //the timestep
 
 for (int i=0;i<k;++i) {
     wait(cfp[i]);
  //calc x displ. and add to x-coord.
  pts_timestep[tn][i].x =pts_timestep[t][i].x+pts_timestep[t][i].vx*timestep+
   .5*pts_timestep[t][i].fx*(timestep*timestep)/pts_timestep[t][i].m;
  //calc new x velocity
  pts_timestep[tn][i].vx =pts_timestep[t][i].vx+pts_timestep[t][i].fx/
                            pts_timestep[t][i].m*timestep;
  //calc y displ. and add to y-coord.
  pts_timestep[tn][i].y =pts_timestep[t][i].y+pts_timestep[t][i].vy*timestep+
   .5*pts_timestep[t][i].fy*(timestep*timestep)/pts_timestep[t][i].m;
  //calc new y velocity
  pts_timestep[tn][i].vy =pts_timestep[t][i].vy+pts_timestep[t][i].fy/
                            pts_timestep[t][i].m*timestep;
  //calc z displ. and add to z-coord.
  pts_timestep[tn][i].z =pts_timestep[t][i].z+pts_timestep[t][i].vz*timestep+
   .5*pts_timestep[t][i].fz*(timestep*timestep)/pts_timestep[t][i].m;
  //calc new z velocity
  pts_timestep[tn][i].vz =pts_timestep[t][i].vz+pts_timestep[t][i].fz/
                            pts_timestep[t][i].m*timestep;
 
 if(debug) {
  cout<<"The New coordinates for point "<<i<<" are: ("<<pts_timestep[tn][i].x
     <<","<<pts_timestep[tn][i].y<<","<<pts_timestep[tn][i].z<<")\n";
  }
 }
}

void printval(promise<void> const & mp,int k,int t,
               ofstream &coorfile, ofstream &trbst) {
 int tn=t+1;
 wait(mp);
 for (int i=0;i<k;i++) {
  coorfile<<pts_timestep[tn][i].x<<","<<pts_timestep[tn][i].y<<","
          <<pts_timestep[tn][i].z<<",";
  trbst<<"v:,"<<pts_timestep[tn][i].vx<<","<<pts_timestep[tn][i].vy<<","
       <<pts_timestep[tn][i].vz<<",";
  trbst<<"f:,"<<pts_timestep[t][i].ft<<",";
 }
 coorfile<<'\n';
 trbst<<'\n';
}

void closefile(ofstream &coorfile,ofstream &trbst,ofstream &notes,
               config_f& param,float ct) {
 notes<<"Here are the notes of the "<<param.output<<" run:\n\n"
   <<"Input file: "<<param.input
   <<"\nNumber of steps: "<<param.steps
   <<"\nTimestep: "<<param.timestep
   <<"\nNumber of threads: "<<param.num_cores
   <<"\nComputation time: "<<ct
   <<"\n\nProgram version: gravity_hpx.3.4\n"; //REMEBER TO UPDATE THIS!!!!!

 coorfile.close();
 trbst.close();
 notes.close();
}

void loadconfig(config_f& param,variables_map& vm) {
 string line;
 ifstream fin(vm["config-file"].as<std::string>());

 while (getline(fin,line)) {
  istringstream sin(line.substr(line.find("=")+1));
  if (line.find("input") != std::string::npos)
   sin>>param.input;
  else if (line.find("output") != std::string::npos)
   sin>>param.output;
  else if (line.find("steps") != std::string::npos)
   sin>>param.steps;
  else if (line.find("timestep") != std::string::npos)
   sin>>param.timestep;
 }
}

void setup(ofstream &coorfile,ofstream &trbst,int k) {
//This section prints out the coordinants and sets up the output files.
 if (debug) cout<<'\n';
 for (int i=0; i<k; ++i) {
  if (debug) {
   cout<<"Point "<<i<<": "<<'('<<pts_timestep[0][i].x<<','
       <<pts_timestep[0][i].y<<','<<pts_timestep[0][i].z<<')'
       <<" "<< pts_timestep[0][i].m<<'\n';
  }
  coorfile<<"Pt:"<<i<<"     ";
  trbst<<"Pt:"<<i<<"                       ";
 }
 cout<<"Number of Points= "<<k<<"\n";
 coorfile<<'\n';
 trbst<<'\n';
 for (int p=0;p<k;p++) {
  coorfile<<"(x,y,z), ";
  trbst<<"v: velx,vely,velz f: ftot, ";
 }
 coorfile<<'\n';
 trbst<<'\n';
 for (int o=0;o<k;o++){
  coorfile<<pts_timestep[0][o].x<<","<<pts_timestep[0][o].y
  <<","<<pts_timestep[0][o].z<<",";
 }
 coorfile<<'\n';
}

#if 0
vector<point> createvecs(ifstream &ist) {
//This section creates and fills the vector to store coordinate and mass data
 vector<point>pts;
  double x;
  double y;
  double z;
  double m;
  double vx;
  double vy;
  double vz;
  while (ist>>x>>y>>z>>m>>vx>>vy>>vz) {
  pts.push_back(point(x,y,z,m,vx,vy,vz));
  if (m/abs(m)==-1) {
   throw invalid_argument("The weight cannot be negative!");
  }
 }
 return pts;
}
#endif

void calc_force(int k,int t,int i) {
 double mt; //the multiple of the masses
 double const g=6.673e-11; //the Gravitational Constant
 double F; //the force
 vector<point> comp2; //vector to copy unit vector info to
 
 for (int l=0;l<k;++l) { //moves through points 1-2, 1-3, etc.
  if (i != l) {
   comp2 = dist(t,i,l); //copy in unit vector data
   mt=pts_timestep[t][l].m*pts_timestep[t][i].m*g;
   F=mt/comp2[0].d;
   
   if (pts_timestep[t][i].x<pts_timestep[t][l].x || 
          pts_timestep[t][i].x==pts_timestep[t][l].x) {
    pts_timestep[t][i].fx+=F*comp2[0].xc; //fx=fx+F*x-component
   }
   if (pts_timestep[t][i].x>pts_timestep[t][l].x) {
    pts_timestep[t][i].fx-=F*comp2[0].xc; //fx=fx-F*x-component
   }
   if (pts_timestep[t][i].y<pts_timestep[t][l].y || 
          pts_timestep[t][i].y==pts_timestep[t][l].y) {
    pts_timestep[t][i].fy+=F*comp2[0].yc; //fx=fx+F*x-component
   }
   if (pts_timestep[t][i].y>pts_timestep[t][l].y) {
    pts_timestep[t][i].fy-=F*comp2[0].yc; //fx=fx-F*x-component
   }
   if (pts_timestep[t][i].z<pts_timestep[t][l].z || 
          pts_timestep[t][i].z==pts_timestep[t][l].z) {
    pts_timestep[t][i].fz+=F*comp2[0].zc; //fx=fx+F*x-component
   }
   if (pts_timestep[t][i].z>pts_timestep[t][l].z) {
    pts_timestep[t][i].fz-=F*comp2[0].zc; //fx=fx-F*x-component
   }
   pts_timestep[t][i].ft=sqrt(pts_timestep[t][i].fx*pts_timestep[t][i].fx+
                              pts_timestep[t][i].fy*pts_timestep[t][i].fy+
                              pts_timestep[t][i].fz*pts_timestep[t][i].fz);
  }
 }
}
