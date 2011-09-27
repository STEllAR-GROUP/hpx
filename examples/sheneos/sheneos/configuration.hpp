//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1221PM)
#define HPX_SHENEOS_CONFIGURATION_AUG_08_2011_1221PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future_value.hpp>
#include <hpx/runtime/components/client_base.hpp>

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
        // create a new partition instance and initialize it synchronously
        configuration(config_data const& data) 
          : base_type(sheneos::stubs::configuration::create_sync(hpx::find_here()))
        {
            init(data);
        }
        configuration(hpx::naming::id_type gid, config_data const& data) 
          : base_type(sheneos::stubs::configuration::create_sync(gid))
        {
            init(data);
        }
        configuration(hpx::naming::id_type gid) 
          : base_type(gid) 
        {}
        configuration()
        {}

        ///////////////////////////////////////////////////////////////////////
        hpx::lcos::future_value<void>
        init_async(config_data const& data)
        {
            return stubs::configuration::init_async(this->gid_, data);
        }

        void init(config_data const& data)
        {
            stubs::configuration::init(this->gid_, data);
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::lcos::future_value<config_data> get_async() const
        {
            return stubs::configuration::get_async(this->gid_);
        }

        config_data get() const
        {
            return stubs::configuration::get(this->gid_);
        }
    };
}

#endif
