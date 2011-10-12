////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach and Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_8B1A4443_7D95_4C0D_9970_7CEA4D049608)
#define HPX_8B1A4443_7D95_4C0D_9970_7CEA4D049608

#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/performance_counters/counters.hpp>

namespace hpx { namespace performance_counters
{
    struct manage_counter
    {
        manage_counter() : counter_(naming::invalid_id) {} 

        ~manage_counter()
        {
            uninstall();
        }

        // create and install a counter which extracts the value from a given 
        // function
        counter_status install(std::string const& name,
            boost::function<boost::int64_t()> const& f, error_code& ec = throws)
        {
            if (0 != counter_) {
                HPX_THROWS_IF(ec, hpx::invalid_status, "manage_counter::install", 
                    "counter has been already installed");
                return status_invalid_data;
            }
 
            info_.fullname_ = name;
            counter_ = create_raw_counter(info_, f, ec);
            return counter_ ? status_valid_data : status_generic_error;
        }

        // create and install a counter which calculates the average rate of 
        // another counter during the given time interval (milliseconds)
        counter_status install(std::string const& name,
            std::string const& base_counter_name, std::size_t base_time_interval, 
            error_code& ec = throws)
        {
            if (0 != counter_) {
                HPX_THROWS_IF(ec, hpx::invalid_status, "manage_counter::install", 
                    "counter has been already installed");
                return status_invalid_data;
            }
 
            info_.fullname_ = name;
            counter_ = create_average_count_counter(info_, base_counter_name, 
                base_time_interval, ec);
            return counter_ ? status_valid_data : status_generic_error;
        }

        // create and install a self-contained counter
        counter_status install(std::string const& name, error_code& ec = throws)
        {
            if (0 != counter_) {
                HPX_THROWS_IF(ec, hpx::invalid_status, "manage_counter::install", 
                    "counter has been already installed");
                return status_invalid_data;
            }
 
            info_.fullname_ = name;
            counter_ = create_counter(info_, ec);
            return counter_ ? status_valid_data : status_generic_error;
        }

        // install an (existing) counter
        counter_status install(naming::id_type const& id, 
            counter_info const& info, error_code& ec = throws)
        {
            if (0 != counter_) {
                HPX_THROWS_IF(ec, hpx::invalid_status, "manage_counter::install", 
                    "counter has been already installed");
                return status_invalid_data;
            }
 
            info_ = info;
            counter_ = id;

            return add_counter(id, info_, ec);
        }

        void uninstall()
        {
            if (counter_)
            {
                error_code ec;
                remove_counter(info_, counter_, ec);
                counter_ = naming::invalid_id;
            }
        }

      private:
        counter_info info_;
        naming::id_type counter_;
    };

    /// Install a new performance counter in a way, which will uninstall it
    /// automatically during shutdown.
    HPX_EXPORT void install_counter(std::string const& name,
        boost::function<boost::int64_t()> const& f, error_code& ec = throws); 

    /// Install a new performance counter in a way, which will uninstall it
    /// automatically during shutdown.
    HPX_EXPORT void install_counter(std::string const& name,
        std::string const& base_counter_name, std::size_t base_time_interval, 
        error_code& ec = throws); 

    HPX_EXPORT void install_counter(std::string const& name,
        error_code& ec = throws); 

    HPX_EXPORT void install_counter(naming::id_type const& id, 
        counter_info const& info, error_code& ec = throws);

    ///////////////////////////////////////////////////////////////////////////
    /// Small data structures holding all data needed to install a counter 
    struct raw_counter_data
    {
        std::string name_;
        boost::function<boost::int64_t()> func_;
    };

    struct average_count_counter_data
    {
        std::string name_;
        std::string base_counter_name_;
        std::size_t base_time_interval_;
    };

    /// Install several new performance counters in a way, which will uninstall 
    /// them automatically during shutdown.
    HPX_EXPORT void install_counters(raw_counter_data const* data, 
        std::size_t count, error_code& ec = throws); 

    /// Install several new performance counters in a way, which will uninstall 
    /// them automatically during shutdown.
    HPX_EXPORT void install_counters(average_count_counter_data const* data, 
        std::size_t count, error_code& ec = throws); 
}}

#endif // HPX_8B1A4443_7D95_4C0D_9970_7CEA4D049608

