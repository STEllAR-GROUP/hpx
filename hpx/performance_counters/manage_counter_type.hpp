////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_F26CC3F9_3E30_4C54_90E0_0CD02146320F)
#define HPX_F26CC3F9_3E30_4C54_90E0_0CD02146320F

#include <hpx/exception.hpp>
#include <hpx/performance_counters/counters.hpp>

namespace hpx { namespace performance_counters
{
    struct manage_counter_type
    {
        manage_counter_type() : status_(status_invalid_data) {}

        ~manage_counter_type()
        {
            uninstall(); 
        }

        counter_status install(std::string const& name, counter_type type,
            error_code& ec = throws)
        {
            if (status_invalid_data != status_) {
                HPX_THROWS_IF(ec, hpx::invalid_status, 
                    "manage_counter_type::install", 
                    "counter type " + name + " has been already installed.");
                return status_invalid_data;
            }

            info_.fullname_ = name;
            info_.type_ = type;
            status_ = add_counter_type(info_, ec);
            return status_;
        }

        void uninstall()
        { 
            if (status_invalid_data != status_)
            {
                error_code ec;
                remove_counter_type(info_, ec); // ignore errors
            }
        }

      private:
        counter_status status_;
        counter_info info_;
    };

#if HPX_AGAS_VERSION > 0x10
    HPX_EXPORT void install_counter_type(std::string const& name,
        counter_type type, error_code& ec = throws); 
#endif
}}

#endif // HPX_F26CC3F9_3E30_4C54_90E0_0CD02146320F

