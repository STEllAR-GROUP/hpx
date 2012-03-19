//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include "read_values.hpp"

#include <boost/assert.hpp>

///////////////////////////////////////////////////////////////////////////////
// Interpolation helper functions related to hdf5
namespace interpolate3d
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
    boost::uint64_t extract_data_range (std::string const& datafilename,
        char const* name, double& minval, double& maxval, double& delta,
        std::size_t start, std::size_t end)
    {
        try {
            using namespace H5;

            // Turn off the auto-printing when failure occurs
            Exception::dontPrint();

            H5File file(datafilename, H5F_ACC_RDONLY);
            DataSet dataset = file.openDataSet(name);    // name of data to read
            DataSpace dataspace = dataset.getSpace();

            // verify number of dimensions
            BOOST_ASSERT(dataspace.getSimpleExtentNdims() == 1);

            // Get the dimension size of each dimension in the dataspace.
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, NULL);
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
    inline void
    read_values(H5::DataSet& dataset, H5::DataSpace& data_space,
        dimension const& dimx, dimension const& dimy, dimension const& dimz,
        double* values)
    {
        using namespace H5;

        // Define hyperslab for file based data
        hsize_t data_offset[dimension::dim] = {
            dimx.offset_, dimy.offset_, dimz.offset_
        };
        hsize_t data_count[dimension::dim] = {
            dimx.count_, dimy.count_, dimz.count_
        };
        data_space.selectHyperslab(H5S_SELECT_SET, data_count, data_offset);

        // memory dataspace
        DataSpace mem_space (dimension::dim, data_count);

        // Define hyperslab for data in memory
        hsize_t mem_offset[dimension::dim] = { 0, 0, 0 };
        mem_space.selectHyperslab(H5S_SELECT_SET, data_count, mem_offset);

        // read data to memory
        dataset.read(values, PredType::NATIVE_DOUBLE, mem_space, data_space);
    }

    ///////////////////////////////////////////////////////////////////////////
    void extract_data(std::string const& datafilename, char const* name,
        double* values, dimension const& dimx, dimension const& dimy,
        dimension const& dimz)
    {
        try {
            using namespace H5;

            // Turn off the auto-printing when failure occurs
            Exception::dontPrint();

            H5File file(datafilename, H5F_ACC_RDONLY);
            DataSet dataset = file.openDataSet(name);
            DataSpace dataspace = dataset.getSpace();

            // verify number of dimensions
            BOOST_ASSERT(dataspace.getSimpleExtentNdims() == dimension::dim);

            // read the data subset
            read_values(dataset, dataspace, dimx, dimy, dimz, values);
        }
        catch (H5::Exception const& e) {
            HPX_THROW_EXCEPTION(hpx::no_success, "extract_data",
                e.getDetailMsg());
        }
    }
}

