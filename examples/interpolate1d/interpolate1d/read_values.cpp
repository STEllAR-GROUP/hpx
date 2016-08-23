//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/assert.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

#include "read_values.hpp"

///////////////////////////////////////////////////////////////////////////////
// Interpolation helper functions related to hdf5
namespace interpolate1d
{
    ///////////////////////////////////////////////////////////////////////////
    inline void
    read_values(H5::DataSet& dataset, H5::DataSpace& data_space,
        hsize_t offset, hsize_t count, double* values)
    {
        using namespace H5;

        // Define hyperslab for file based data
        hsize_t data_offset[1] = { offset };
        hsize_t data_count[1] = { count };
        data_space.selectHyperslab(H5S_SELECT_SET, data_count, data_offset);

        // memory dataspace
        hsize_t mem_dims[1] = { count };
        DataSpace mem_space (1, mem_dims);

        // Define hyperslab for data in memory
        hsize_t mem_offset[1] = { 0 };
        hsize_t mem_count[1] = { count };
        mem_space.selectHyperslab(H5S_SELECT_SET, mem_count, mem_offset);

        // read data to memory
        dataset.read(values, PredType::NATIVE_DOUBLE, mem_space, data_space);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::uint64_t extract_data_range (std::string const& datafilename,
        double& minval, double& maxval, double& delta,
        std::size_t start, std::size_t end)
    {
        try {
            using namespace H5;

            // Turn off the auto-printing when failure occurs
            Exception::dontPrint();

            H5File file(datafilename, H5F_ACC_RDONLY);
            DataSet dataset = file.openDataSet("x");    // name of data to read
            DataSpace dataspace = dataset.getSpace();

            // number of dimensions
            int numdims = dataspace.getSimpleExtentNdims();

            if (numdims != 1)
            {
                HPX_THROW_EXCEPTION(hpx::no_success, "extract_data_range",
                    "number of dimensions was not 1");
            }

            // Get the dimension size of each dimension in the dataspace.
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, nullptr);
            if (end == std::size_t(-1))
                end = dims[0];

            read_values(dataset, dataspace, start, 1, &minval);
            read_values(dataset, dataspace, end-1, 1, &maxval);
            read_values(dataset, dataspace, start+1, 1, &delta);

            delta -= minval;
            return dims[0];     // return size of dataset
        }
        catch (H5::Exception const& e) {
            HPX_THROW_EXCEPTION(hpx::no_success, "extract_data_range",
                e.getDetailMsg());
        }
        return 0;   // keep compiler happy
    }

    ///////////////////////////////////////////////////////////////////////////
    void extract_data(std::string const& datafilename, double* values,
      std::size_t offset, std::size_t count)
    {
        try {
            using namespace H5;

            // Turn off the auto-printing when failure occurs
            Exception::dontPrint();

            H5File file(datafilename, H5F_ACC_RDONLY);
            DataSet dataset = file.openDataSet("sine"); // name of data to read
            DataSpace dataspace = dataset.getSpace();

            // number of dimensions
            int numdims = dataspace.getSimpleExtentNdims();

            if (numdims != 1)
            {
                HPX_THROW_EXCEPTION(hpx::no_success, "extract_data",
                    "number of dimensions was not 1");
            }

            // Get the dimension size of each dimension in the dataspace.
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, nullptr);

            read_values(dataset, dataspace, offset, count, values);
        }
        catch (H5::Exception const& e) {
            HPX_THROW_EXCEPTION(hpx::no_success, "extract_data",
                e.getDetailMsg());
        }
    }
}

