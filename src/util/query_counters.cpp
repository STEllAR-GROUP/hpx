//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/util/unlock_guard.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/get_config_entry.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <boost/format.hpp>
#include <boost/thread/locks.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace hpx { namespace util
{
    query_counters::query_counters(std::vector<std::string> const& names,
            boost::int64_t interval, std::string const& dest, std::string const& form,
            std::vector<std::string> const& shortnames, bool csv_header)
      : names_(names), destination_(dest), format_(form),
            counter_shortnames_(shortnames), csv_header_(csv_header),
        timer_(boost::bind(&query_counters::evaluate, this_()),
            boost::bind(&query_counters::terminate, this_()),
            interval*1000, "query_counters", true)
    {
        // add counter prefix, if necessary
            for (std::string& name : names_) {
                performance_counters::ensure_counter_prefix(name);
            }
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
        boost::unique_lock<mutex_type> l(mtx_);

        std::vector<std::string> names;
        std::swap(names, names_);

        names_.reserve(names.size());
        if (ids_.empty())
        {
            using util::placeholders::_1;
            using util::placeholders::_2;

            performance_counters::discover_counter_func func(
                util::bind(&query_counters::find_counter, this, _1, _2));

            ids_.reserve(names.size());
            uoms_.reserve(names.size());
            for (std::string& name : names)
            {
                // do INI expansion on counter name
                util::expand(name);

                // find matching counter type
                {
                    hpx::util::unlock_guard<boost::unique_lock<mutex_type> > ul(l);
                    performance_counters::discover_counter_type(name, func,
                        performance_counters::discover_counters_full);
                }
            }
        }

        HPX_ASSERT(ids_.size() == names_.size());
        HPX_ASSERT(ids_.size() == uoms_.size());
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

    void query_counters::stop_evaluating_counters()
    {
        timer_.stop();
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

    template <typename Stream>
    void query_counters::print_name_csv(Stream& out, std::string const& name)
    {
        out << performance_counters::remove_counter_prefix(name);
    }

    template <typename Stream>
    void query_counters::print_value_csv(Stream& out,
        performance_counters::counter_value const& value)
    {
        error_code ec(lightweight);
        double val = value.get_value<double>(ec);
        if(!ec) {
            out << val;
        }
        else {
            out << "invalid";
        }
    }

    template <typename Stream>
    void query_counters::print_name_csv_short(Stream& out, std::string const& name)
    {
        out << name;
    }

    bool query_counters::evaluate()
    {
        bool reset = false;
        if (get_config_entry("hpx.print_counter.reset", "0") == "1")
            reset = true;

        return evaluate_counters(reset);
    }

    void query_counters::terminate()
    {
        std::vector<naming::id_type> ids;
        {
            boost::lock_guard<mutex_type> l(mtx_);
            // give up control over all performance counters
            std::swap(ids, ids_);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    void query_counters::start_counters(error_code& ec)
    {
        bool has_been_started = false;
        {
            boost::lock_guard<mutex_type> l(mtx_);
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

    void query_counters::stop_counters(error_code& ec)
    {
        bool has_been_started = false;
        {
            boost::lock_guard<mutex_type> l(mtx_);
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
        wait_all(stopped);

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

    void query_counters::reset_counters(error_code& ec)
    {
        bool has_been_started = false;
        {
            boost::lock_guard<mutex_type> l(mtx_);
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
            boost::lock_guard<mutex_type> l(mtx_);
            has_been_started = !ids_.empty();
            destination_is_cout = destination_ == "cout";
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
            boost::lock_guard<mutex_type> l(mtx_);
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

        std::ostringstream output;
        if (description)
            output << description << std::endl;

//         // wait for all values to be returned
//         wait_all(values);

        // Output the performance counter value.
        if (csv_header_ == true) {
            if(format_ == "csv") {
                for (std::size_t i = 0; i < names_.size(); ++i)
                {
                    print_name_csv(output, names_[i]);
                    if (i != names_.size()-1)
                        output << ",";
                }
                output << "\n";
            }

            if(format_ == "csv-short") {
                for (std::size_t i = 0; i < counter_shortnames_.size(); ++i)
                {
                    print_name_csv_short(output, counter_shortnames_[i]);
                    if (i != counter_shortnames_.size()-1)
                        output << ",";
                }
                output << "\n";
            }
            csv_header_ = false;
        }

        if (format_ == "csv" || format_ == "csv-short") {
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                print_value_csv(output, values[i].get());
                if (i != values.size()-1)
                    output << ",";
            }
            output << "\n";
        }
        else {
            for (std::size_t i = 0; i < values.size(); ++i)
                print_value(output, names_[i], values[i].get(), uoms_[i]);
        }

        if (destination_is_cout) {
            std::cout << output.str() << std::flush;
        }
        else {
            std::ofstream out(destination_.c_str(), std::ofstream::app);
            out << output.str();
        }

        return true;
    }
}
}

