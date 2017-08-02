//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013      Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/activate_counters.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/util/unwrap.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <boost/format.hpp>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace util
{
    activate_counters::activate_counters(std::vector<std::string> const& names)
      : names_(names)
    {
        start_counters();
    }

    activate_counters::~activate_counters()
    {
        stop_counters();
    }

    bool activate_counters::find_counter(
        performance_counters::counter_info const& info, error_code& ec)
    {
        naming::id_type id = performance_counters::get_counter(info.fullname_, ec);
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "activate_counters::find_counter",
                boost::str(boost::format(
                    "unknown performance counter: '%1%' (%2%)") %
                    info.fullname_ % ec.get_message()));
            return false;
        }

        names_.push_back(info.fullname_);
        ids_.push_back(id);
        uoms_.push_back(info.unit_of_measure_);
        return true;
    }

    void activate_counters::find_counters()
    {
        std::vector<std::string> names;
        std::swap(names, names_);

        names_.reserve(names.size());
        if (ids_.empty())
        {
            using util::placeholders::_1;
            using util::placeholders::_2;

            performance_counters::discover_counter_func func(
                util::bind(&activate_counters::find_counter, this, _1, _2));

            ids_.reserve(names.size());
            uoms_.reserve(names.size());
            for (std::string& name : names)
            {
                // do INI expansion on counter name
                util::expand(name);

                // find matching counter type
                performance_counters::discover_counter_type(name, func,
                    performance_counters::discover_counters_full);
            }
        }

        HPX_ASSERT(ids_.size() == names_.size());
        HPX_ASSERT(ids_.size() == uoms_.size());
    }

    ///////////////////////////////////////////////////////////////////////////
    void activate_counters::start_counters(error_code& ec)
    {
        using performance_counters::stubs::performance_counter;

        // add counter prefix, if necessary
        for (std::string& name : names_)
            performance_counters::ensure_counter_prefix(name);

        find_counters();

        // Query the performance counters.
        std::vector<future<bool> > started;

        started.reserve(ids_.size());
        for (std::size_t i = 0; i != ids_.size(); ++i)
            started.push_back(performance_counter::start(launch::async, ids_[i]));

        // wait for all counters to be started
        wait_all(started);

        for (future<bool>& f : started)
        {
            if (f.has_exception())
            {
                if (&ec == &hpx::throws)
                {
                    f.get();
                }
                else
                {
                    ec = make_error_code(f.get_exception_ptr());
                }
                return;
            }
        }
    }

    void activate_counters::stop_counters(error_code& ec)
    {
        if (ids_.empty())
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "activate_counters::stop_counters",
                "The counters to be evaluated have not been initialized yet");
            return;
        }

        // Query the performance counters.
        using performance_counters::stubs::performance_counter;
        std::vector<future<bool> > stopped;

        stopped.reserve(ids_.size());
        for (std::size_t i = 0; i != ids_.size(); ++i)
            stopped.push_back(performance_counter::stop(launch::async, ids_[i]));

        // wait for all counters to be started
        wait_all(stopped);

        ids_.clear();      // give up control over all performance counters

        for (future<bool>& f : stopped)
        {
            if (f.has_exception())
            {
                if (&ec == &hpx::throws)
                {
                    f.get();
                }
                else
                {
                    ec = make_error_code(f.get_exception_ptr());
                }
                return;
            }
        }
    }

    void activate_counters::reset_counters(error_code& ec)
    {
        if (ids_.empty())
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status, "activate_counters::reset_counters",
                "The counters to be evaluated have not been initialized yet");
            return;
        }

        // Query the performance counters.
        using performance_counters::stubs::performance_counter;
        std::vector<future<void> > reset;

        reset.reserve(ids_.size());
        for (std::size_t i = 0; i != ids_.size(); ++i)
            reset.push_back(performance_counter::reset(launch::async, ids_[i]));

        // wait for all counters to be started
        wait_all(reset);

        for (future<void>& f : reset)
        {
            if (f.has_exception())
            {
                if (&ec == &hpx::throws)
                {
                    f.get();
                }
                else
                {
                    ec = make_error_code(f.get_exception_ptr());
                }
                return;
            }
        }
    }

    std::vector<future<performance_counters::counter_value> >
    activate_counters::evaluate_counters(launch::async_policy, bool reset,
        error_code& ec)
    {
        std::vector<future<performance_counters::counter_value> > values;

        if (ids_.empty())
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "activate_counters::evaluate_counters_async",
                "The counters to be evaluated have not been initialized yet");
            return values;
        }

        values.reserve(ids_.size());
        using hpx::performance_counters::stubs::performance_counter;
        for (std::size_t i = 0; i != ids_.size(); ++i)
        {
            values.push_back(performance_counter::get_value(
                launch::async, ids_[i], reset));
        }
        return values;
    }

    std::vector<performance_counters::counter_value>
    activate_counters::evaluate_counters(launch::sync_policy, bool reset,
        error_code& ec)
    {
        std::vector<future<performance_counters::counter_value> >
            futures = evaluate_counters(launch::async, reset, ec);

        return util::unwrap(futures);
    }
}}

