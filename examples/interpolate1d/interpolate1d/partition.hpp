//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARTITION_AUG_04_2011_0251PM)
#define HPX_PARTITION_AUG_04_2011_0251PM

#include <hpx/hpx.hpp>
#include <hpx/include/client.hpp>

#include <cstddef>
#include <string>

#include "server/partition.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace interpolate1d
{
    class partition
      : public hpx::components::client_base<partition, server::partition>
    {
    private:
        typedef hpx::components::client_base<partition, server::partition>
            base_type;

    public:
        // create a new partition instance and initialize it synchronously
        partition(std::string datafilename, dimension const& dim,
                std::size_t num_nodes)
          : base_type(hpx::new_<server::partition>(hpx::find_here()))
        {
            init(datafilename, dim, num_nodes);
        }

        partition(hpx::id_type id, std::string datafilename,
                dimension const& dim, std::size_t num_nodes)
          : base_type(hpx::new_<server::partition>(id))
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
            typedef server::partition::init_action init_action;
            return hpx::async(init_action(), this->get_id(), datafilename,
                dim, num_nodes);
        }

        void init(std::string datafilename, dimension const& dim,
            std::size_t num_nodes)
        {
            init_async(datafilename, dim, num_nodes).get();
        }

        // ask this partition to interpolate, note that value must be in the
        // range valid for this partition
        hpx::lcos::future<double>
        interpolate_async(double value) const
        {
            typedef server::partition::interpolate_action interpolate_action;
            return hpx::async(interpolate_action(), this->get_id(), value);
        }

        double interpolate(double value) const
        {
            return interpolate_async(value).get();
        }
    };
}

#endif


