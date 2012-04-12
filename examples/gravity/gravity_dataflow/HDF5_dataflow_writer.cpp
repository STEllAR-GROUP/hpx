///////////////////////////////////////////////////////////////////////////////
/////////////////////    HDF5 Dataflow Writer    //////////////////////////////
//
//This program is designed to write the output file of Gravity.  It writes
//the file in a HDF5 format.
//
///////////////////////////////////////////////////////////////////////////////
//Copyright (c) 2012 Adrian Serio
//
//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
///////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <string>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <hpx/lcos/async.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <H5Cpp.h>
#include "gravity_dataflow.hpp"

using namespace std;
using namespace H5;

using hpx::lcos::future;
using hpx::lcos::wait;

void printval (Vector_container pts,config_f const & param,uint64_t k,
                uint64_t t) {
 if (t%param.print==0) {
  uint64_t width=log10(param.steps)+1;
  string a=boost::lexical_cast<string>(t);
  string frm=boost::str(boost::format("%%s_%%0%dd.h5") %width);
  string name=boost::str(boost::format(frm) %param.output %a);
  const H5std_string FILE_NAME(name);
  const H5std_string DATASET_NAME("Coordinates");
  H5File file(FILE_NAME, H5F_ACC_TRUNC);
  hsize_t fdim[]={k,11};
  DataSpace fspace(2,fdim);
  DataSet dataset=file.createDataSet(DATASET_NAME, PredType::NATIVE_DOUBLE,
                                     fspace);
  dataset.write(&pts[0],PredType::NATIVE_DOUBLE);
  dataset.close();
  file.close();
 }
}

void printdebug(Vector_container pts,uint64_t k,ofstream &coorfile, 
                 ofstream &trbst) {
 if (debug) {
  for (uint64_t i=0;i<k;i++) {
   coorfile<<pts[i].x<<","<<pts[i].y<<","<<pts[i].z<<",";
   trbst<<"v:,"<<pts[i].vx<<","<<pts[i].vy<<","<<pts[i].vz<<",";
   trbst<<"f:,"<<pts[i].ft<<",";
 }
 coorfile<<'\n';
 trbst<<'\n';
 }
}

void printfinalcoord (Vector_container const &pts,config_f& param, uint64_t k) { 
                          //Not done with this function
 string j=param.output+"_finalcoord.h5";
 const H5std_string FILE_NAME(j);
 const H5std_string DATASET_NAME("Coordinates");
 H5File file(FILE_NAME, H5F_ACC_TRUNC); //Create new file
 
 //Define a memory space
 hsize_t mdim[]={k,11}; //memory dimentions
 DataSpace mspace(2,mdim); //memory dataspace
 
 //Define a hyperslab
 hsize_t count[2];
 hsize_t offset[2];
 hsize_t block[2];
 count[0]=1;
 count[1]=1;
 offset[0]=0;
 offset[1]=0;
 block[0]=k;
 block[1]=7;
 
 mspace.selectHyperslab(H5S_SELECT_SET,count,offset,NULL,block); //create
                                                                 //hyperslab
 //Define new file space
 hsize_t fdim[]={k,7}; //new file dimentions
 DataSpace fspace(2,fdim); //new file dataspace
 DataSet dataset=file.createDataSet(DATASET_NAME, PredType::NATIVE_DOUBLE,
                                    fspace); //new file dataset
 dataset.write(&pts[0],PredType::NATIVE_DOUBLE,mspace,
               fspace); //write memory-hyperslab to file
 dataset.close();
 file.close();
}
