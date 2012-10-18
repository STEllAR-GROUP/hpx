//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_INTERPOLATE3D_CONFIGURATION_AUG_07_2011_0703PM)
#define HPX_INTERPOLATE3D_CONFIGURATION_AUG_07_2011_0703PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/include/client.hpp>

#include "stubs/configuration.hpp"

namespace interpolate3d
{
    ///////////////////////////////////////////////////////////////////////////
    class configuration
      : public hpx::components::client_base<
            configuration, interpolate3d::stubs::configuration>
    {
    private:
        typedef hpx::components::client_base<
            configuration, interpolate3d::stubs::configuration> base_type;

    public:
        // create a new partition instance and initialize it synchronously
        configuration(std::string const& datafilename,
                std::string const& symbolic_name, std::size_t num_instances)
          : base_type(interpolate3d::stubs::configuration::create(hpx::find_here()))
        {
            init(datafilename, symbolic_name, num_instances);
        }
        configuration(hpx::naming::id_type gid, std::string const& datafilename,
                std::string const& symbolic_name, std::size_t num_instances)
          : base_type(interpolate3d::stubs::configuration::create(gid))
        {
            init(datafilename, symbolic_name, num_instances);
        }
        configuration(hpx::naming::id_type gid)
          : base_type(gid)
        {}
        configuration()
        {}

        ///////////////////////////////////////////////////////////////////////
        hpx::lcos::future<void>
        init_async(std::string const& datafile,
            std::string const& symbolic_name, std::size_t num_instances)
        {
            return stubs::configuration::init_async(this->get_gid(), datafile,
                symbolic_name, num_instances);
        }

        void init(std::string const& datafile, std::string const& symbolic_name,
            std::size_t num_instances)
        {
            stubs::configuration::init(this->get_gid(), datafile, symbolic_name,
                num_instances);
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::lcos::future<config_data> get_async() const
        {
            return stubs::configuration::get_async(this->get_gid());
        }

        config_data get() const
        {
            return stubs::configuration::get(this->get_gid());
        }
    };
}

#endif
