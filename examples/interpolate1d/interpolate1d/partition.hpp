//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARTITION_AUG_04_2011_0251PM)
#define HPX_PARTITION_AUG_04_2011_0251PM

#include <hpx/hpx.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/include/client.hpp>

#include <string>

#include "stubs/partition.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace interpolate1d
{
    class partition
      : public hpx::components::client_base<
            partition, interpolate1d::stubs::partition>
    {
    private:
        typedef hpx::components::client_base<
            partition, interpolate1d::stubs::partition> base_type;

    public:
        // create a new partition instance and initialize it synchronously
        partition(std::string datafilename, dimension const& dim,
                std::size_t num_nodes)
          : base_type(interpolate1d::stubs::partition::create(hpx::find_here()))
        {
            init(datafilename, dim, num_nodes);
        }
        partition(hpx::naming::id_type gid, std::string datafilename,
                dimension const& dim, std::size_t num_nodes)
          : base_type(interpolate1d::stubs::partition::create(gid))
        {
            init(datafilename, dim, num_nodes);
        }
        partition(hpx::naming::id_type gid)
          : base_type(gid)
        {}

        // initialize this partition
        hpx::lcos::future<void>
        init_async(std::string datafilename, dimension const& dim,
            std::size_t num_nodes)
        {
            return stubs::partition::init_async(this->get_id(), datafilename,
                dim, num_nodes);
        }

        void init(std::string datafilename, dimension const& dim,
            std::size_t num_nodes)
        {
            stubs::partition::init(this->get_id(), datafilename, dim, num_nodes);
        }

        // ask this partition to interpolate, note that value must be in the
        // range valid for this partition
        hpx::lcos::future<double>
        interpolate_async(double value)
        {
            return stubs::partition::interpolate_async(this->get_id(), value);
        }

        double interpolate(double value)
        {
            return stubs::partition::interpolate(this->get_id(), value);
        }
    };
}

#endif


