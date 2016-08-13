//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/assert.hpp>

#include <cstdint>
#include <string>

#include "read_values.hpp"

///////////////////////////////////////////////////////////////////////////////
// Interpolation helper functions related to HDF5.
namespace sheneos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    inline void
    read_values(H5::DataSet& dataset, H5::DataSpace& data_space,
        hsize_t offset, hsize_t count, double* values)
    {
        using namespace H5;

        // Define the hyperslab for file based data.
        hsize_t data_offset[1] = { offset };
        hsize_t data_count[1] = { count };
        data_space.selectHyperslab(H5S_SELECT_SET, data_count, data_offset);

        // Memory dataspace.
        hsize_t mem_dims[1] = { count };
        DataSpace mem_space (1, mem_dims);

        // Define the hyperslab for data in memory.
        hsize_t mem_offset[1] = { 0 };
        hsize_t mem_count[1] = { count };
        mem_space.selectHyperslab(H5S_SELECT_SET, mem_count, mem_offset);

        // Read data to memory.
        dataset.read(values, PredType::NATIVE_DOUBLE, mem_space, data_space);
    }

} // sheneos::detail

    ///////////////////////////////////////////////////////////////////////////
    void extract_data(std::string const& datafilename, char const* name,
        double* values, hsize_t offset, hsize_t count)
    {
        try {
            using namespace H5;

            // Turn off auto-printing on failure.
            Exception::dontPrint();

            // Try to open the file.
            H5File file(datafilename, H5F_ACC_RDONLY);

            // Try to open the specified dataset.
            DataSet dataset = file.openDataSet(name);
            DataSpace dataspace = dataset.getSpace();

            // Verify number of dimensions.
            HPX_ASSERT(dataspace.getSimpleExtentNdims() == 1);

            // Read the data subset.
            detail::read_values(dataset, dataspace, offset, count, values);
        }
        catch (H5::Exception const& e) {
            HPX_THROW_EXCEPTION(hpx::no_success, "sheneos::extract_data",
                e.getDetailMsg());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    std::uint64_t extract_data_range(std::string const& datafilename,
        char const* name, double& minval, double& maxval, double& delta,
        std::size_t start, std::size_t end)
    {
        try {
            using namespace H5;

            // Turn off auto-printing on failure.
            Exception::dontPrint();

            // Try to open the file.
            H5File file(datafilename, H5F_ACC_RDONLY);

            // Try to open the specified dataset.
            DataSet dataset = file.openDataSet(name);
            DataSpace dataspace = dataset.getSpace();

            // Verify number of dimensions
            HPX_ASSERT(dataspace.getSimpleExtentNdims() == 1);

            // Get the size of each dimension in the dataspace.
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, nullptr);
            if (end == std::size_t(-1))
                end = dims[0];

            // Read the minimum and maximum values.
            detail::read_values(dataset, dataspace, start, 1, &minval);
            detail::read_values(dataset, dataspace, end - 1, 1, &maxval);

            // Read the delta value.
            detail::read_values(dataset, dataspace, start + 1, 1, &delta);
            delta -= minval;

            // Return size of dataset.
            return dims[0];
        }
        catch (H5::Exception e) {
            std::string msg = e.getDetailMsg().c_str();
            HPX_THROW_EXCEPTION(hpx::no_success,
                "sheneos::extract_data_range", msg);
        }

        // This return statement keeps the compiler from whining.
        return 0;
    }

namespace detail
{

    ///////////////////////////////////////////////////////////////////////////
    inline void
    read_values(H5::DataSet& dataset, H5::DataSpace& data_space,
        dimension const& dimx, dimension const& dimy, dimension const& dimz,
        double* values)
    {
        using namespace H5;

        // Define the hyperslab for file based data.
        hsize_t data_offset[dimension::dim] = {
            dimx.offset_, dimy.offset_, dimz.offset_
        };
        hsize_t data_count[dimension::dim] = {
            dimx.count_, dimy.count_, dimz.count_
        };
        data_space.selectHyperslab(H5S_SELECT_SET, data_count, data_offset);

        // Memory dataspace.
        DataSpace mem_space (dimension::dim, data_count);

        // Define the hyperslab for data in memory.
        hsize_t mem_offset[dimension::dim] = { 0, 0, 0 };
        mem_space.selectHyperslab(H5S_SELECT_SET, data_count, mem_offset);

        // Read data to memory.
        dataset.read(values, PredType::NATIVE_DOUBLE, mem_space, data_space);
    }

} // sheneos::detail

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

            // Verify number of dimensions.
            HPX_ASSERT(dataspace.getSimpleExtentNdims() == dimension::dim);

            // Read the data subset.
            detail::read_values(dataset, dataspace, dimx, dimy, dimz, values);
        }
        catch (H5::Exception const& e) {
            HPX_THROW_EXCEPTION(hpx::no_success, "sheneos::extract_data",
                e.getDetailMsg());
        }
    }
}

