//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_INTERPOLATE1D_AUG_04_2011_0340PM)
#define HPX_INTERPOLATE1D_AUG_04_2011_0340PM

#include <hpx/hpx.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "partition.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace interpolate1d
{
    // This class encapsulates N partitions and dispatches requests based on
    // the given values to get interpolated results for.
    class HPX_COMPONENT_EXPORT interpolate1d
    {
    public:
        // Create a new interpolation instance and initialize it synchronously.
        // Passing -1 as the second argument creates exactly one partition
        // instance on each available locality.
        interpolate1d(std::string datafilename,
            std::size_t num_instances = std::size_t(-1));

        // Return the interpolated  function value for the given argument. This
        // function dispatches to the proper partition for the actual
        // interpolation.
        hpx::lcos::future<double>
        interpolate_async(double value) const
        {
            return get_partition(value).interpolate_async(value);
        }

        double interpolate(double value) const
        {
            return get_partition(value).interpolate(value);
        }

    private:
        // map the given value to the gid of the partition responsible for the
        // interpolation
        partition get_partition(double value) const;

        // initialize the partitions and store the mappings
        void fill_partitions(std::string const& datafilename,
            hpx::future<std::vector<partition> > && partitions);

    private:
        std::vector<partition> partitions_;
        std::uint64_t num_elements_;
        double minval_, delta_;
    };
}

#endif


