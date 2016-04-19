//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_INTERPOLATE1D_AUG_04_2011_0340PM)
#define HPX_INTERPOLATE1D_AUG_04_2011_0340PM

#include <hpx/hpx.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/components/distributing_factory/distributing_factory.hpp>

#include <string>
#include <vector>

#include "stubs/partition.hpp"

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
        interpolate_async(double value)
        {
            return stubs::partition::interpolate_async(get_id(value), value);
        }

        double interpolate(double value)
        {
            return stubs::partition::interpolate(get_id(value), value);
        }

    private:
        // map the given value to the gid of the partition responsible for the
        // interpolation
        hpx::naming::id_type get_id(double value);

        // initialize the partitions and store the mappings
        typedef hpx::components::distributing_factory distributing_factory;
        typedef distributing_factory::async_create_result_type
            async_create_result_type;

        void fill_partitions(std::string const& datafilename,
            async_create_result_type future);

    private:
        std::vector<hpx::naming::id_type> partitions_;
        boost::uint64_t num_elements_;
        double minval_, delta_;
    };
}

#endif


