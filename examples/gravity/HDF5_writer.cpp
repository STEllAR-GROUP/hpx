///////////////////////////////////////////////////////////////////////////////
//////////////////////////    HDF5 Writer    //////////////////////////////////
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
#include <stdlib.h>
#include <string>

#include <boost/lexical_cast.hpp>

#include <hpx/lcos/async.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <H5Cpp.h>
#include "gravity.hpp"

using namespace std;
using namespace H5;

using hpx::lcos::promise;
using hpx::lcos::wait;

void printval (promise<void> const & mp,config_f& param,int k,int t) {
 string a=boost::lexical_cast<string>(t);
 string j=param.output+"_"+a+".h5";
 wait(mp);
 const H5std_string FILE_NAME(j);
 const H5std_string DATASET_NAME("Coordinates");
 H5File file(FILE_NAME, H5F_ACC_TRUNC);
 hsize_t fdim[]={k,15};
 DataSpace fspace(2,fdim);
 DataSet dataset=file.createDataSet(DATASET_NAME, PredType::NATIVE_DOUBLE,
                                    fspace);
 dataset.write(&pts_timestep[t][0],PredType::NATIVE_DOUBLE);
 dataset.close();
 file.close();
}

void printfinalcoord (config_f& param, int k) { ///////Not done with this function
 string j=param.output+"_finalcoord.h5";
 const H5std_string FILE_NAME(j);
 const H5std_string DATASET_NAME("Coordinates");
 H5File file(FILE_NAME, H5F_ACC_TRUNC); //Create new file
 hsize_t fdim[]={k,15}; //new file dimentions
 DataSpace fspace(2,fdim); //new file dataspace
 //DataSet dataset=file.createDataSet(DATASET_NAME, PredType::NATIVE_DOUBLE,
 //                                   fspace); //new dataset
 
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
 
 fspace.selectHyperslab(H5S_SELECT_SET,count,offset,NULL,block);
 DataSet dataset=file.createDataSet(DATASET_NAME, PredType::NATIVE_DOUBLE,
                                    fspace); //new dataset
 dataset.write(&pts_timestep[param.steps][0],PredType::NATIVE_DOUBLE);
 dataset.close();
 file.close();
}
 ///////////////////This BS still doesnt work...hyperslab not recognized////
