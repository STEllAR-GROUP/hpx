///////////////////////////////////////////////////////////////////////////////
//////////////////////////    HDF5 Writer    //////////////////////////////////
//
//This program is designed to write the output file of Gravity.  It writes
//the file in a HDF5 format.
//
///////////////////////////////////////////////////////////////////////////////
//Copyright (c) Adrian Serio
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
 int tn=t+1;
 string a=boost::lexical_cast<string>(t);
 string j=param.output+"_"+a+".h5";
 wait(mp);
 const H5std_string FILE_NAME(j);
 const H5std_string DATASET_NAME("Coordinates");
 H5File file(FILE_NAME, H5F_ACC_TRUNC);
 hsize_t fdim[]={k,11};
 DataSpace fspace(2,fdim);
 DataSet dataset=file.createDataSet(DATASET_NAME, PredType::NATIVE_DOUBLE,
                                    fspace);
 dataset.write(&pts_timestep[tn][0],PredType::NATIVE_DOUBLE);//Willnot print forces
 dataset.close();
 file.close();
}

