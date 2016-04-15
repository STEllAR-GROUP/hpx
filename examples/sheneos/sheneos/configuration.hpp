//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1221PM)
#define HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1221PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/include/client.hpp>

#include <string>

#include "stubs/configuration.hpp"

namespace sheneos
{
    ///////////////////////////////////////////////////////////////////////////
    class configuration
      : public hpx::components::client_base<
            configuration, sheneos::stubs::configuration>
    {
    private:
        typedef hpx::components::client_base<
            configuration, sheneos::stubs::configuration> base_type;

    public:
        /// Create a new partition instance and initialize it synchronously.
        configuration(std::string const& datafilename,
                std::string const& symbolic_name, std::size_t num_instances)
          : base_type(sheneos::stubs::configuration::create(hpx::find_here()))
        {
            init(datafilename, symbolic_name, num_instances);
        }
        configuration(hpx::naming::id_type gid, std::string const& datafilename,
                std::string const& symbolic_name, std::size_t num_instances)
          : base_type(sheneos::stubs::configuration::create(gid))
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
            return stubs::configuration::init_async(this->get_id(), datafile,
                symbolic_name, num_instances);
        }

        void init(std::string const& datafile, std::string const& symbolic_name,
            std::size_t num_instances)
        {
            stubs::configuration::init(this->get_id(), datafile, symbolic_name,
                num_instances);
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::lcos::future<config_data> get_async() const
        {
            return stubs::configuration::get_async(this->get_id());
        }

        config_data get() const
        {
            return stubs::configuration::get(this->get_id());
        }
    };
}

#endif

