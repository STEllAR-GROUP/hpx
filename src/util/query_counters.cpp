//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <boost/assert.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include <iostream>
#include <fstream>

namespace hpx { namespace util
{
    query_counters::query_counters(std::vector<std::string> const& names,
            boost::int64_t interval, std::string const& dest)
      : names_(names), destination_(dest),
        timer_(boost::bind(&query_counters::evaluate, this_()),
            boost::bind(&query_counters::terminate, this_()),
            interval*1000, "query_counters", true)
    {
        // add counter prefix, if necessary
        BOOST_FOREACH(std::string& name, names_)
            performance_counters::ensure_counter_prefix(name);
    }

    bool query_counters::find_counter(
        performance_counters::counter_info const& info, error_code& ec)
    {
        naming::id_type id = performance_counters::get_counter(info.fullname_, ec);
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, bad_parameter,
                "query_counters::find_counter",
                boost::str(boost::format(
                    "unknown performance counter: '%1%' (%2%)") %
                    info.fullname_ % ec.get_message()));
            return false;
        }

        names_.push_back(info.fullname_);
        ids_.push_back(id);

//         using performance_counters::stubs::performance_counter;
//         performance_counters::counter_info info =
//             performance_counter::get_info(id);

        uoms_.push_back(info.unit_of_measure_);
        return true;
    }

    void query_counters::find_counters()
    {
        mutex_type::scoped_lock l(mtx_);

        std::vector<std::string> names;
        std::swap(names, names_);

        names_.reserve(names.size());
        if (ids_.empty())
        {
            using HPX_STD_PLACEHOLDERS::_1;
            using HPX_STD_PLACEHOLDERS::_2;

            HPX_STD_FUNCTION<performance_counters::discover_counter_func> func(
                HPX_STD_BIND(&query_counters::find_counter, this, _1, _2));

            ids_.reserve(names.size());
            uoms_.reserve(names.size());
            BOOST_FOREACH(std::string& name, names)
            {
                // do INI expansion on counter name
                util::expand(name);

                // find matching counter type
                {
                    hpx::util::scoped_unlock<mutex_type::scoped_lock> ul(l);
                    performance_counters::discover_counter_type(name, func,
                        performance_counters::discover_counters_full);
                }
            }
        }

        BOOST_ASSERT(ids_.size() == names_.size());
        BOOST_ASSERT(ids_.size() == uoms_.size());
    }

    void query_counters::start()
    {
        find_counters();

        for (std::size_t i = 0; i != ids_.size(); ++i)
        {
            // start the performance counter
            using performance_counters::stubs::performance_counter;
            performance_counter::start(ids_[i]);
        }

        // this will invoke the evaluate function for the first time
        timer_.start();
    }

    template <typename Stream>
    void query_counters::print_value(Stream& out, std::string const& name,
        performance_counters::counter_value const& value, std::string const& uom)
    {
        error_code ec(lightweight);        // do not throw
        double val = value.get_value<double>(ec);

#ifdef HPX_HAVE_APEX
        apex::sample_value(name.c_str(), val);
#endif

        out << performance_counters::remove_counter_prefix(name) << ",";
        out << value.count_ << ",";
        if (!ec) {
            double elapsed = static_cast<double>(value.time_) * 1e-9;
            out << boost::str(boost::format("%.6f") % elapsed)
                << ",[s]," << val;
            if (!uom.empty())
                out << ",[" << uom << "]";
            out << "\n";
        }
        else {
            out << "invalid\n";
        }
    }

    bool query_counters::evaluate()
    {
        return evaluate_counters();
    }

    void query_counters::terminate()
    {
        mutex_type::scoped_lock l(mtx_);
        ids_.clear();      // give up control over all performance counters
    }

    ///////////////////////////////////////////////////////////////////////////
    void query_counters::start_counters(error_code& ec)
    {
        bool has_been_started = false;
        {
            mutex_type::scoped_lock l(mtx_);
            has_been_started = !ids_.empty();
        }

        if (!has_been_started)
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "query_counters::start_counters",
                "The counters to be evaluated have not been initialized yet");
            return;
        }

        // Query the performance counters.
        using performance_counters::stubs::performance_counter;
        std::vector<future<bool> > started;

        started.reserve(ids_.size());
        for (std::size_t i = 0; i != ids_.size(); ++i)
            started.push_back(performance_counter::start_async(ids_[i]));

        // wait for all counters to be started
        wait_all(started, ec);
    }

    void query_counters::stop_counters(error_code& ec)
    {
        bool has_been_started = false;
        {
            mutex_type::scoped_lock l(mtx_);
            has_been_started = !ids_.empty();
        }

        if (!has_been_started)
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "query_counters::stop_counters",
                "The counters to be evaluated have not been initialized yet");
            return;
        }

        // Query the performance counters.
        using performance_counters::stubs::performance_counter;
        std::vector<future<bool> > stopped;

        stopped.reserve(ids_.size());
        for (std::size_t i = 0; i != ids_.size(); ++i)
            stopped.push_back(performance_counter::stop_async(ids_[i]));

        // wait for all counters to be started
        wait_all(stopped, ec);
    }

    void query_counters::reset_counters(error_code& ec)
    {
        bool has_been_started = false;
        {
            mutex_type::scoped_lock l(mtx_);
            has_been_started = !ids_.empty();
        }

        if (!has_been_started)
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "query_counters::reset_counters",
                "The counters to be evaluated have not been initialized yet");
            return;
        }

        // Query the performance counters.
        using performance_counters::stubs::performance_counter;
        std::vector<future<void> > reset;

        reset.reserve(ids_.size());
        for (std::size_t i = 0; i != ids_.size(); ++i)
            reset.push_back(performance_counter::reset_async(ids_[i]));

        // wait for all counters to be started
        wait_all(reset, ec);
    }

    bool query_counters::evaluate_counters(bool reset,
        char const* description, error_code& ec)
    {
        if (timer_.is_terminated())
        {
            // just do nothing as we're about to terminate the application
            ids_.clear();       // free all held performance counters
            return false;
        }

        bool has_been_started = false;
        bool destination_is_cout = false;
        {
            mutex_type::scoped_lock l(mtx_);
            has_been_started = !ids_.empty();
            destination_is_cout = (destination_ == "cout") ? true : false;
        }

        if (!has_been_started)
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "query_counters::evaluate",
                "The counters to be evaluated have not been initialized yet");
            return false;
        }

        std::vector<id_type> ids;
        {
            mutex_type::scoped_lock l(mtx_);
            ids = ids_;
        }

        if (ids.empty())
            return false;

        // Query the performance counters.
        using performance_counters::stubs::performance_counter;
        std::vector<future<performance_counters::counter_value> > values;

        values.reserve(ids.size());
        for (std::size_t i = 0; i != ids.size(); ++i)
            values.push_back(performance_counter::get_value_async(ids[i], reset));

        util::osstream output;
        if (description)
            output << description << std::endl;

//         // wait for all values to be returned
//         wait_all(values, ec);

        // Output the performance counter value.
        for (std::size_t i = 0; i < values.size(); ++i)
            print_value(output, names_[i], values[i].get(), uoms_[i]);

        if (destination_is_cout) {
            std::cout << util::osstream_get_string(output) << std::flush;
        }
        else {
            std::ofstream out(destination_, std::ios_base::app);
            out << util::osstream_get_string(output);
        }

        return true;
    }
}}

