///////////////////////////////////////////////////////////////////////////////
/////////////////////////////     GRAVITY VD    ///////////////////////////////
//This program calculates the effect of gravity on a set of points in three
//dimentions. It uses a "coordinate file" to define the points and gives the
//user ability to adjust the number of iterations and time that passes for each
//iteration. These files and values are changed in the configuraion file,
//in this case "gravconfig.conf".
//
//Problem: Updating to use dataflows
//
//Remember to change the program version when program is updated!
///////////////////////////////////////////////////////////////////////////////
//Copyright (c) 2011-2012 Adrian Serio
//
//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
///////////////////////////////////////////////////////////////////////////////
//Units are in m, kg, s, and therefore N.
///////////////////////////////////////////////////////////////////////////////
#include<hpx/hpx.hpp>
#include<hpx/hpx_fwd.hpp>
#include<hpx/hpx_init.hpp>
#include<hpx/components/dataflow/dataflow.hpp> ///!!!///
#include<hpx/runtime/actions/plain_action.hpp>
#include<hpx/runtime/components/plain_component_factory.hpp>
#include<hpx/util/high_resolution_timer.hpp>
#include<hpx/lcos/async.hpp>

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

#include "gravity_dataflow.hpp"

using namespace std;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;
using hpx::async;
using hpx::lcos::future;
using hpx::lcos::wait;
using hpx::lcos::dataflow; ///!!!!////
using hpx::lcos::dataflow_base; ///!!!!////
using hpx::util::high_resolution_timer;
using hpx::init;
using hpx::finalize;
using hpx::find_here;
using hpx::get_os_thread_count;

///////////////////////////////////////////////////////////////////////////////
//Forward Declarations
//vector<future<void> > calc(uint64_t k,uint64_t t);
Vector_container calc(Vector_container pts,uint64_t k,uint64_t t);
vector<components> dist(Vector_container &pts,uint64_t t,uint64_t i,uint64_t l);
Vector_container calc_move(Vector_container pts,config_f const & param,
                uint64_t k,uint64_t t);
void printval(Vector_container pts,config_f const & param,uint64_t k,
               uint64_t t);///!!!!//
void printdebug(Vector_container pts,uint64_t k,ofstream &coorfile,
                 ofstream &trbst); ///!!!///
void printfinalcoord(Vector_container const &pts,config_f& param,uint64_t k);
void closefile(ofstream &notes,config_f& param,float ct);
void closedebugfile(ofstream &coorfile,ofstream &trbst); //closefile for debug
void loadconfig(config_f& param,variables_map& vm);
void setup(Vector_container pts,ofstream &coorfile,ofstream &trbst,uint64_t k);
Vector_container createvecs(config_f& param);
void calc_force(Vector_container pts,uint64_t k,uint64_t t,uint64_t i);
Vector_container identity(Vector_container pts);
///////////////////////////////////////////////////////////////////////////////
//Action Declarations
HPX_PLAIN_ACTION(calc);
HPX_PLAIN_ACTION(calc_force);
HPX_PLAIN_ACTION(move);
HPX_PLAIN_ACTION(identity, id_action);
HPX_PLAIN_ACTION(print);

///////////////////////////////////////////////////////////////////////////////
//Global Variables
bool debug;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
  {
     config_f param;
     ofstream coorfile; //optional output file
     ofstream trbst;    //optional output file
     string j;          // Buffer
     uint64_t k=0;      //Number of points in simulation
     float ct=0;        //Computation time

     debug=vm.count("debug");
     cout <<"Config File Location: "<< vm["config-file"].as<std::string>()
          << "\n";
     param.num_cores=get_os_thread_count();

    //This section opens an input file
     loadconfig(param,vm);              //load the configuraton file
     ifstream ist(param.input.c_str()); //Name of input file ="param.input"
     if(!ist) {
      cout<<"Cannot open input file!\n";
     }

     if (debug) {
      // This section creates and opens the coordinate output file
      j=param.output+".cf.txt";
      coorfile.open(j.c_str());

      //This section opens the trb output file
      j=param.output+".trb.txt";
      trbst.open(j.c_str());
     }

    //This section opens and creates a documentation output file
     j=param.output+".notes.txt";
     ofstream notes;
     notes.open(j.c_str());

    //This section sets up the rest of the program
//     pts_timestep = vector<vector<point> >(param.steps+1,createvecs(param));
//     k=pts_timestep[0].size();
     Vector_container pts (createvecs(param));
//     k=param.steps+1;
     k=param.steps;
     cout<<"Number of Points= "<<k<<"\n";
     if (debug) { setup(pts,coorfile,trbst,k); } // sets up the output files
     dataflow_base<Vector_container >move = dataflow<id_action>(find_here(),pts);
     high_resolution_timer ht;

     //This section outlines the program
     for (uint64_t t=0;t<param.steps;t++) {
      if (debug) cout<<"\nFor step "<<t<<":\n";
#if 0
      cfp=calc(k,t);  //This is the calculation of force
      mp=async<move_action>(find_here(),cfp,param,k,t);
      printval(mp,param,k,t,coorfile,trbst); //Writes output
#endif
      dataflow_base<Vector_container > calculate=
                                   dataflow<calc_action>(find_here(),move,k,t);
                                   //This is the calculation of force
      move=dataflow<move_action>(find_here(),calculate,param,k,t);
      dataflow_base<void>printval=
             dataflow<print_action>(find_here(),move,param,k,t);
             //Writes output ?????
      if(debug) { printdebug(pts,k,coorfile,trbst); }

     }
     Vector_container result = move.get_future().get();
     printfinalcoord(pts,param,k);
     ct=ht.elapsed();
     cout<<"Computation Time: "<<ct<<" [s]\n";
     closefile(notes,param,ct); //normal closefile
     if (debug) { closedebugfile(coorfile,trbst); } //debug closefile
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

//vector<future<void> > calc(uint64_t k,uint64_t t) {
Vector_container calc(Vector_container pts,uint64_t k,uint64_t t) {

 vector<future<void> > cfp;
 for (uint64_t i=0;i<k;++i) { //moves through the points ie 1, 2, etc.
  cfp.push_back(async<calc_force_action>(find_here(),pts,k,t,i));
  wait(cfp[i]); /////////////!!!!!!!!!!!!!!!!!!!!!!//////////////////////////
 }
// return cfp;
 return pts;
}

vector<components> dist(Vector_container &pts,uint64_t t,uint64_t i,uint64_t l) {
 double rx;  //distance in the x direction
 double ry;  //distance in the y direction
 double rz;  //distance in the z direction
 double r2t; //distance between the two points squared
 double rt;  //distance between the two points

#if 0
 rx=abs(pts_timestep[t][l].x-pts_timestep[t][i].x);
 ry=abs(pts_timestep[t][l].y-pts_timestep[t][i].y);
 rz=abs(pts_timestep[t][l].z-pts_timestep[t][i].z);
 r2t=(rx*rx)+(ry*ry)+(rz*rz);
 rt=sqrt(r2t);
#endif

 rx=abs(pts[l].x-pts[i].x);   //////!!!!!////
 ry=abs(pts[l].y-pts[i].y);   //
 rz=abs(pts[l].z-pts[i].z);   //
 r2t=(rx*rx)+(ry*ry)+(rz*rz); //
 rt=sqrt(r2t);                //////!!!!!!//////

vector<components>comp(1);

 comp[0].d=r2t;
 comp[0].xc=rx/rt;
 comp[0].yc=ry/rt;
 comp[0].zc=rz/rt;

 return comp;
}

//void move(vector<future<void> >const& cfp,config_f const & param,
//                uint64_t k,uint64_t t) {
Vector_container calc_move(Vector_container pts,config_f const & param,
                uint64_t k,uint64_t t) {
// uint64_t tn=t+1;
 double timestep=param.timestep; //the timestep

 for (uint64_t i=0;i<k;++i) {
#if 0
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
 #endif

  //calc x displ. and add to x-coord.
  pts[i].x =pts[i].x+pts[i].vx*timestep+
             .5*pts[i].fx*(timestep*timestep)/pts[i].m;
  //calc new x velocity
  pts[i].vx =pts[i].vx+pts[i].fx/pts[i].m*timestep;
  //calc y displ. and add to y-coord.
  pts[i].y =pts[i].y+pts[i].vy*timestep+
             .5*pts[i].fy*(timestep*timestep)/pts[i].m;
  //calc new y velocity
  pts[i].vy =pts[i].vy+pts[i].fy/pts[i].m*timestep;
  //calc z displ. and add to z-coord.
  pts[i].z =pts[i].z+pts[i].vz*timestep+
             .5*pts[i].fz*(timestep*timestep)/pts[i].m;
  //calc new z velocity
  pts[i].vz =pts[i].vz+pts[i].fz/pts[i].m*timestep;

#if 0
 if(debug) {
  cout<<"The New coordinates for point "<<i<<" are: ("<<pts_timestep[tn][i].x
     <<","<<pts_timestep[tn][i].y<<","<<pts_timestep[tn][i].z<<")\n";
  }
#endif
 if(debug) {
  cout<<"The New coordinates for point "<<i<<" are: ("<<pts[i].x
     <<","<<pts[i].y<<","<<pts[i].z<<")\n";
  }
 }
 return pts;
}

void closefile(ofstream &notes,config_f& param,float ct) {
 notes<<"Here are the notes of the "<<param.output<<" run:\n\n"
   <<"Input file: "<<param.input
   <<"\nNumber of steps: "<<param.steps
   <<"\nTimestep: "<<param.timestep
   <<"\nNumber of threads: "<<param.num_cores
   <<"\nComputation time: "<<ct
   <<"\n\nProgram version: gravity_dataflow.3.5\n"; //REMEBER TO UPDATE THIS!!!!!

 notes.close();
}

void closedebugfile(ofstream &coorfile,ofstream &trbst) { //debug overload
 coorfile.close();
 trbst.close();
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
  else if (line.find("print") != std::string::npos)
   sin>>param.print;
 }
}

void setup(Vector_container pts,ofstream &coorfile,ofstream &trbst,uint64_t k) {
//This section prints out the coordinants and sets up the output files.
 cout<<'\n';
 for (uint64_t i=0; i<k; ++i) {
//  cout<<"Point "<<i<<": "<<'('<<pts_timestep[0][i].x<<','
//      <<pts_timestep[0][i].y<<','<<pts_timestep[0][i].z<<')'
//      <<" "<< pts_timestep[0][i].m<<'\n';
  cout<<"Point "<<i<<": "<<'('<<pts[i].x<<','
      <<pts[i].y<<','<<pts[i].z<<')'
      <<" "<< pts[i].m<<'\n';

  coorfile<<"Pt:"<<i<<"     ";
  trbst<<"Pt:"<<i<<"                       ";
 }
 coorfile<<'\n';
 trbst<<'\n';
 for (uint64_t p=0;p<k;p++) {
  coorfile<<"(x,y,z), ";
  trbst<<"v: velx,vely,velz f: ftot, ";
 }
 coorfile<<'\n';
 trbst<<'\n';
 for (uint64_t o=0;o<k;o++){
//  coorfile<<pts_timestep[0][o].x<<","<<pts_timestep[0][o].y
//  <<","<<pts_timestep[0][o].z<<",";
  coorfile<<pts[o].x<<","<<pts[o].y<<","<<pts[o].z<<",";
 }
 coorfile<<'\n';
}

//void calc_force(uint64_t k,uint64_t t,uint64_t i) {
void calc_force(Vector_container pts, uint64_t k,uint64_t t,uint64_t i) {
 double mt; //the multiple of the masses
 double const g=6.673e-11; //the Gravitational Constant
 double F; //the force
 vector<components> comp2; //vector to copy unit vector info to

 for (uint64_t l=0;l<k;++l) { //moves through points 1-2, 1-3, etc.
  if (i != l) {
#if 0
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
#endif
   comp2 = dist(pts,t,i,l); //copy in unit vector data
   mt=pts[l].m*pts[i].m*g;
   F=mt/comp2[0].d;
   if (pts[i].x<pts[l].x || pts[i].x==pts[l].x) {
    pts[i].fx+=F*comp2[0].xc; //fx=fx+F*x-component
   }
   if (pts[i].x>pts[l].x) {
    pts[i].fx-=F*comp2[0].xc; //fx=fx-F*x-component
   }
   if (pts[i].y<pts[l].y || pts[i].y==pts[l].y) {
    pts[i].fy+=F*comp2[0].yc; //fx=fx+F*x-component
   }
   if (pts[i].y>pts[l].y) {
    pts[i].fy-=F*comp2[0].yc; //fx=fx-F*x-component
   }
   if (pts[i].z<pts[l].z || pts[i].z==pts[l].z) {
    pts[i].fz+=F*comp2[0].zc; //fx=fx+F*x-component
   }
   if (pts[i].z>pts[l].z) {
    pts[i].fz-=F*comp2[0].zc; //fx=fx-F*x-component
   }
   pts[i].ft=sqrt(pts[i].fx*pts[i].fx+pts[i].fy*pts[i].fy+pts[i].fz*pts[i].fz);


  }
 }
}

Vector_container identity(Vector_container pts) {
 return pts;
}
