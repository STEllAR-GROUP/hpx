//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Create some 3D test data for the interpolate3d example

#include <H5Cpp.h>
#include <vector>
#include <cmath>

using namespace H5;

int main()
{
    std::size_t const num_points_x = 100;
    std::size_t const num_points_y = 100;
    std::size_t const num_points_z = 100;

    std::vector<double> data_x;
    std::vector<double> data_y;
    std::vector<double> data_z;
    data_x.reserve(num_points_x+1);
    data_y.reserve(num_points_y+1);
    data_z.reserve(num_points_z+1);

    std::vector<double> values;
    values.reserve((num_points_x+1) * (num_points_y+1) * (num_points_z+1));

    for (double x = -5; x <= 5; x += 10./num_points_x)
        data_x.push_back(x);
    for (double y = -5; y <= 5; y += 10./num_points_y)
        data_y.push_back(y);
    for (double z = -5; z <= 5; z += 10./num_points_z)
        data_z.push_back(z);

    for (double x = -5; x <= 5; x += 10./num_points_x) {
        for (double y = -5; y <= 5; y += 10./num_points_y) {
            for (double z = -5; z <= 5; z += 10./num_points_z) {
                values.push_back(std::exp(-x*x - y*y - z*z));
            }
        }
    }

    try {
        // Turn off the auto-printing when failure occurs
        Exception::dontPrint();

        // create a new file, truncate any existing file
        H5File file("gauss.h5", H5F_ACC_TRUNC);

        // Define the size of the data array and create the data space for
        // fixed sized data array
        hsize_t dims_x[1] = { num_points_x+1 };
        DataSpace dataspace_x(1, dims_x);

        hsize_t dims_y[1] = { num_points_y+1 };
        DataSpace dataspace_y(1, dims_y);

        hsize_t dims_z[1] = { num_points_z+1 };
        DataSpace dataspace_z(1, dims_z);

        FloatType datatype(PredType::NATIVE_DOUBLE);
        datatype.setOrder(H5T_ORDER_LE);

        // Create a new dataset within the file using defined dataspace and
        // datatype and default dataset creation properties.
        DataSet dataset_data_x = file.createDataSet("x", datatype, dataspace_x);
        DataSet dataset_data_y = file.createDataSet("y", datatype, dataspace_y);
        DataSet dataset_data_z = file.createDataSet("z", datatype, dataspace_z);

        // Write the data to the dataset using default memory space, file
        // space, and transfer properties.
        dataset_data_x.write(&*data_x.begin(), PredType::NATIVE_DOUBLE);
        dataset_data_y.write(&*data_y.begin(), PredType::NATIVE_DOUBLE);
        dataset_data_z.write(&*data_z.begin(), PredType::NATIVE_DOUBLE);

        // Create a new dataset within the file using defined dataspace and
        // datatype and default dataset creation properties.
        hsize_t dims[3] = { num_points_x+1, num_points_y+1, num_points_z+1 };
        DataSpace dataspace(3, dims);
        DataSet dataset_values = file.createDataSet("gauss", datatype, dataspace);

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
