//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Create some test data for the interpolate1d example

#include <H5Cpp.h>

#include <cmath>
#include <vector>

using namespace H5;

int main()
{
    double const pi = 4*std::atan(1.);
    std::size_t const num_points = 36000;

    std::vector<double> data;
    std::vector<double> values;
    data.reserve(num_points+1);
    values.reserve(num_points+1);

    for (std::size_t i = 0; i <= num_points; ++i) {
        data.push_back(2.*pi*i/num_points);
        values.push_back(std::sin(2.*pi*i/num_points));
    }

    try {
        // Turn off the auto-printing when failure occurs
        Exception::dontPrint();

        // create a new file, truncate any existing file
        H5File file("sine.h5", H5F_ACC_TRUNC);

        // Define the size of the data array and create the data space for
        // fixed sized data array
        hsize_t dimsf[1];              // dataset dimensions
        dimsf[0] = num_points+1;
        DataSpace dataspace(1, dimsf);

        FloatType datatype(PredType::NATIVE_DOUBLE);
        datatype.setOrder(H5T_ORDER_LE);

        // Create a new dataset within the file using defined dataspace and
        // datatype and default dataset creation properties.
        DataSet dataset_data = file.createDataSet("x", datatype, dataspace);

        // Write the data to the dataset using default memory space, file
        // space, and transfer properties.
        dataset_data.write(&*data.begin(), PredType::NATIVE_DOUBLE);

        // Create a new dataset within the file using defined dataspace and
        // datatype and default dataset creation properties.
        DataSet dataset_values = file.createDataSet("sine", datatype, dataspace);

        // Write the data to the dataset using default memory space, file
        // space, and transfer properties.
        dataset_values.write(&*values.begin(), PredType::NATIVE_DOUBLE);
    }
    catch(FileIException const& error) {
        // catch failure caused by the H5File operations
        error.printError();
        return -1;
    }
    catch(DataSetIException const& error) {
        // catch failure caused by the DataSet operations
        error.printError();
        return -1;
    }
    catch(DataSpaceIException const& error) {
        // catch failure caused by the DataSpace operations
        error.printError();
        return -1;
    }
    catch(DataTypeIException const& error) {
        // catch failure caused by the DataSpace operations
        error.printError();
        return -1;
    }
    return 0;
}
