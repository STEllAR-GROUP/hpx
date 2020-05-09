//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <H5Cpp.h>

#include <cstddef>
#include <cstdint>
#include <string>

#if !defined(H5_HAVE_THREADSAFE)
#error    "This example requires that the HDF5 API is thread-safe. Please provide a suitable version of HDF5."
#endif

///////////////////////////////////////////////////////////////////////////////
// Interpolation helper functions related to hdf5
namespace interpolate1d
{
    // extract the lower and upper bounds, etc.
    std::uint64_t extract_data_range (std::string const& datafilename,
        double& minval, double& maxval, double& delta, std::size_t start = 0,
        std::size_t end = std::size_t(-1));

    // extract the actual data slice
    void extract_data(std::string const& datafilename, double* values,
      std::size_t offset, std::size_t count);
}


