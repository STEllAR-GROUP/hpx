////////////////////////////////////////////////////////////////////////////////
///////////////////////////      HDF5 Reader    ////////////////////////////////
//
//This program is designed to read in an HDF5 file produced by Coordinate
//Builder. It is used to setup the points in Gravity.
//
////////////////////////////////////////////////////////////////////////////////
//Copyright (c) Adrian Serio
//
//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdlib>

#include <H5Cpp.h>
#include "gravity.hpp"

using namespace std;
using namespace H5;

const H5std_string DATASET_NAME("Coordinates"); //Coord. Builder names the
                                                //dataset "Coordinates"

vector<point> createvecs (config_f& param) {
 const H5std_string FILE_NAME(param.input);
 H5File file(FILE_NAME, H5F_ACC_RDONLY); //Open HDF5 file
 DataSet dataset=file.openDataSet(DATASET_NAME); //Open dataset
 DataSpace dataspace=dataset.getSpace(); //Get the dataspace
 hsize_t dims_out[2];
 uint64_t ndims=0;
 ndims=dataspace.getSimpleExtentDims(dims_out,NULL); //Get dimentions

 //Define a memory block
 hsize_t dimsm[3];
 dimsm[0]=dims_out[0];
 dimsm[1]=dims_out[1];
 dimsm[2]=1;
 DataSpace memspace(3,dimsm);
 
 //Define a buffer
 
// double data_out[dims_out[0]][dims_out[1]][1];

 typedef double array_type[7][1];

 double (* data_out)[7][1] = new array_type [dims_out[0]];

 dataset.read(data_out,PredType::NATIVE_DOUBLE,memspace,dataspace); //Read data
 
 vector<point> pts;
 for (hsize_t i=0;i<dims_out[0];i++) {
  pts.push_back(point(data_out[i][0][0],data_out[i][1][0],data_out[i][2][0],
                      data_out[i][3][0],data_out[i][4][0],data_out[i][5][0],
                      data_out[i][6][0]));
 }

 delete data_out; 

 return pts;
} 
