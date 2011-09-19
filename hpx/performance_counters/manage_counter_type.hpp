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
        manage_counter_type(counter_info const& info) 
          : info_(info), status_(status_invalid_data) 
        {}

        ~manage_counter_type()
        {
            uninstall(); 
        }

        counter_status install(error_code& ec = throws)
        {
            if (status_invalid_data != status_) {
                HPX_THROWS_IF(ec, hpx::invalid_status, 
                    "manage_counter_type::install", 
                    "counter type " + info_.fullname_ + 
                    " has been already installed.");
                return status_invalid_data;
            }

            return status_ = add_counter_type(info_, ec);
        }

        void uninstall(error_code& ec = throws)
        { 
            if (status_invalid_data != status_)
                remove_counter_type(info_, ec); // ignore errors
        }

    private:
        counter_status status_;
        counter_info info_;
    };

#if HPX_AGAS_VERSION > 0x10
    /// Install a new performance counter type in a way, which will uninstall 
    /// it automatically during shutdown.
    HPX_EXPORT void install_counter_type(std::string const& name,
        counter_type type, error_code& ec = throws); 

    /// A small data structure holding all data needed to install a counter type
    struct counter_type_data
    {
        std::string name_;          ///< Name of teh counter type
        counter_type type_;         ///< Type of the counter instances of this 
                                    ///< counter type
        std::string helptext_;      ///< Longer descriptive text explaining the 
                                    ///< counter type
        boost::uint32_t version_;   ///< Version of this counter type definition 
                                    ///< (default: 0x01000000)
    };

    /// Install several new performance counter types in a way, which will 
    /// uninstall them automatically during shutdown.
    HPX_EXPORT void install_counter_types(counter_type_data const* data,
        std::size_t count, error_code& ec = throws); 
#endif
}}

#endif // HPX_F26CC3F9_3E30_4C54_90E0_0CD02146320F

