//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_INTERPOLATE3D_READ_VALUES_AUG_05_2011_1056AM)
#define HPX_INTERPOLATE3D_READ_VALUES_AUG_05_2011_1056AM

#include <H5Cpp.h>
#include <boost/cstdint.hpp>

#include "dimension.hpp"

///////////////////////////////////////////////////////////////////////////////
// Interpolation helper functions related to hdf5
namespace interpolate3d
{
    // extract the lower and upper bounds, etc.
    boost::uint64_t extract_data_range (std::string const& datafilename,
        char const* name, double& minval, double& maxval, double& delta,
        std::size_t start = 0, std::size_t end = std::size_t(-1));

    ///////////////////////////////////////////////////////////////////////////
    // extract the actual 3d data slice
    void extract_data(std::string const& datafilename, char const* name,
        double* values, dimension const& dimx, dimension const& dimy,
        dimension const& dimz);
}

#endif

