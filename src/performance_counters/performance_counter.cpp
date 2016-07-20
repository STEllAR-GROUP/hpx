//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/bind.hpp>

#include <hpx/performance_counters/performance_counter.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    performance_counter::performance_counter(std::string const& name)
      : base_type(performance_counters::get_counter_async(name))
    {}

    performance_counter::performance_counter(std::string const& name,
        hpx::id_type const& locality)
    {
        HPX_ASSERT(naming::is_locality(locality));

        counter_path_elements p;
        get_counter_type_path_elements(name, p);

        std::string full_name;
        p.parentinstanceindex_ = naming::get_locality_id_from_id(locality);
        get_counter_name(p, full_name);

        this->base_type::reset(performance_counters::get_counter_async(full_name));
    }

    ///////////////////////////////////////////////////////////////////////////
    future<counter_info> performance_counter::get_info() const
    {
        return stubs::performance_counter::get_info_async(get_id());
    }
    counter_info performance_counter::get_info_sync(error_code& ec) const
    {
        return stubs::performance_counter::get_info(get_id(), ec);
    }

    future<counter_value> performance_counter::get_counter_value(bool reset)
    {
        return stubs::performance_counter::get_value_async(get_id(), reset);
    }
    counter_value performance_counter::get_counter_value_sync(bool reset,
        error_code& ec)
    {
        return stubs::performance_counter::get_value(get_id(), reset, ec);
    }

    future<counter_value> performance_counter::get_counter_value() const
    {
        return stubs::performance_counter::get_value_async(get_id(), false);
    }
    counter_value performance_counter::get_counter_value_sync(error_code& ec) const
    {
        return stubs::performance_counter::get_value(get_id(), false, ec);
    }

    future<counter_values_array>
    performance_counter::get_counter_values_array(bool reset)
    {
        return stubs::performance_counter::get_values_array_async(get_id(), reset);
    }
    counter_values_array
    performance_counter::get_counter_values_array_sync(bool reset, error_code& ec)
    {
        return stubs::performance_counter::get_values_array(get_id(), reset, ec);
    }

    future<counter_values_array>
    performance_counter::get_counter_values_array() const
    {
        return stubs::performance_counter::get_values_array_async(get_id(), false);
    }
    counter_values_array
    performance_counter::get_counter_values_array_sync(error_code& ec) const
    {
        return stubs::performance_counter::get_values_array(get_id(), false, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    future<bool> performance_counter::start()
    {
        return stubs::performance_counter::start_async(get_id());
    }
    bool performance_counter::start_sync(error_code& ec)
    {
        return stubs::performance_counter::start(get_id(), ec);
    }

    future<bool> performance_counter::stop()
    {
        return stubs::performance_counter::stop_async(get_id());
    }
    bool performance_counter::stop_sync(error_code& ec)
    {
        return stubs::performance_counter::stop(get_id(), ec);
    }

    future<void> performance_counter::reset()
    {
        return stubs::performance_counter::reset_async(get_id());
    }
    void performance_counter::reset_sync(error_code& ec)
    {
        stubs::performance_counter::reset(get_id(), ec);
    }

    ///
    future<std::string> performance_counter::get_name() const
    {
        return lcos::make_future<std::string>(
            get_info(),
            [](counter_info && info) -> std::string
            {
                return info.fullname_;
            });
    }

    std::string performance_counter::get_name_sync() const
    {
        return get_name().get();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Return all counters matching the given name (with optional wildcards).
    std::vector<performance_counter>
    discover_counters(std::string const& name, error_code& ec)
    {
        std::vector<performance_counter> counters;

        std::vector<counter_info> infos;
        counter_status status = discover_counter_type(name, infos,
            discover_counters_full, ec);
        if (!status_is_valid(status) || ec)
            return counters;

        try {
            counters.reserve(infos.size());
            for (counter_info const& info : infos)
            {
                performance_counter counter(info.fullname_);
                counters.push_back(counter);
            }
        }
        catch (hpx::exception const& e) {
            HPX_RETHROWS_IF(ec, e, "discover_counters");
        }

        return counters;
    }
}}
