//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/runtime/get_thread_name.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/apex.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <boost/format.hpp>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
#include <map>
#include <ittnotify.h>
#endif

namespace hpx { namespace util
{
    query_counters::query_counters(std::vector<std::string> const& names,
            std::vector<std::string> const& reset_names,
            std::int64_t interval, std::string const& dest, std::string const& form,
            std::vector<std::string> const& shortnames, bool csv_header)
      : names_(names), reset_names_(reset_names),
        destination_(dest), format_(form),
        counter_shortnames_(shortnames), csv_header_(csv_header),
        timer_(util::bind(&query_counters::evaluate, this_()),
            util::bind(&query_counters::terminate, this_()),
            interval*1000, "query_counters", true)
    {
        // add counter prefix, if necessary
        for (std::string& name : names_)
        {
            performance_counters::ensure_counter_prefix(name);
        }
        for (std::string& name : reset_names_)
        {
            performance_counters::ensure_counter_prefix(name);
        }
    }

    void query_counters::find_counters()
    {
        if (!names_.empty())
            counters_.add_counters(names_);
        if (!reset_names_.empty())
            counters_.add_counters(reset_names_, true);

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        if (use_ittnotify_api)
        {
            typedef std::map<std::string, util::itt::counter>::value_type
                value_type;

            for (auto const& info : counters_.get_counter_infos())
            {
                std::string real_name =
                    performance_counters::remove_counter_prefix(info.fullname_);
                itt_counters_.insert(
                    value_type(info.fullname_, util::itt::counter(
                        real_name.c_str(), hpx::get_thread_name().c_str(),
                        __itt_metadata_double
                    ))
                );
            }
        }
#endif
    }

    void query_counters::start()
    {
        find_counters();

        counters_.start(launch::sync);

        // this will invoke the evaluate function for the first time
        timer_.start();
    }

    void query_counters::stop_evaluating_counters()
    {
        timer_.stop();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Stream>
    void query_counters::print_name_csv(Stream& out, std::string const& name)
    {
        std::string s = performance_counters::remove_counter_prefix(name);
        if (s.find_first_of(",") != std::string::npos)
            out << "\"" << s << "\"";
        else
            out << s;
    }

    template <typename Stream>
    void query_counters::print_value(Stream* out, std::string const& name,
        performance_counters::counter_value const& value, std::string const& uom)
    {
        error_code ec(lightweight);        // do not throw
        double val = value.get_value<double>(ec);

        if(!ec) {
#ifdef HPX_HAVE_APEX
            apex::sample_value(name.c_str(), val);
#elif HPX_HAVE_ITTNOTIFY != 0
            if (use_ittnotify_api)
            {
                auto it = itt_counters_.find(name);
                if (it != itt_counters_.end())
                {
                    (*it).second.set_value(val);
                }
            }
#endif

            if (out == nullptr)
                return;

            print_name_csv(*out, name);
            *out  << "," << value.count_ << ",";

            double elapsed = static_cast<double>(value.time_) * 1e-9;
            *out << boost::str(boost::format("%.6f") % elapsed)
                << ",[s]," << val;
            if (!uom.empty())
                *out << ",[" << uom << "]";
            *out << "\n";
        }
        else {
            if (out != nullptr)
                *out << "invalid\n";
        }
    }

    template <typename Stream>
    void query_counters::print_value(Stream* out, std::string const& name,
        performance_counters::counter_values_array const& value,
        std::string const& uom)
    {
        if (out == nullptr)
            return;

        error_code ec(lightweight);        // do not throw

        print_name_csv(*out, name);
        *out << "," << value.count_ << ",";

        double elapsed = static_cast<double>(value.time_) * 1e-9;
        *out << boost::str(boost::format("%.6f") % elapsed) << ",[s],";

        bool first = true;
        for (std::int64_t val : value.values_)
        {
            if (!first)
                *out << ':';
            first = false;
            *out << val;
        }

        if (!uom.empty())
            *out << ",[" << uom << "]";
        *out << "\n";
    }

    template <typename Stream>
    void query_counters::print_value_csv(Stream* out, std::string const& name,
        performance_counters::counter_value const& value)
    {
        error_code ec(lightweight);
        double val = value.get_value<double>(ec);

        if(!ec) {
#ifdef HPX_HAVE_APEX
            apex::sample_value(name.c_str(), val);
#elif HPX_HAVE_ITTNOTIFY != 0
            if (use_ittnotify_api)
            {
                auto it = itt_counters_.find(name);
                if (it != itt_counters_.end())
                {
                    (*it).second.set_value(val);
                }
                return;
            }
#endif
            if (out == nullptr)
                return;

            *out << val;
        }
        else {
            if (out != nullptr)
                *out << "invalid";
        }
    }

    template <typename Stream>
    void query_counters::print_value_csv(Stream* out, std::string const&,
        performance_counters::counter_values_array const& value)
    {
        if (out == nullptr)
            return;

        bool first = true;
        for (std::int64_t val : value.values_)
        {
            if (!first)
                *out << ':';
            first = false;
            *out << val;
        }
    }

    template <typename Stream>
    void query_counters::print_name_csv_short(Stream& out, std::string const& name)
    {
        out << name;
    }

    template <typename Stream>
    void query_counters::print_headers(Stream& output,
        std::vector<performance_counters::counter_info> const& infos)
    {
        if (csv_header_) {
            if (format_ == "csv")
            {
                // first print raw value counters
                bool first = true;
                for (std::size_t i = 0; i != infos.size(); ++i)
                {
                    if (infos[i].type_ != performance_counters::counter_raw)
                        continue;
                    if (!first)
                        output << ",";
                    first = false;
                    print_name_csv(output, infos[i].fullname_);
                }

                // now print array value counters
                for (std::size_t i = 0; i != infos.size(); ++i)
                {
                    if (infos[i].type_ != performance_counters::counter_histogram)
                        continue;
                    if (!first)
                        output << ",";
                    first = false;
                    print_name_csv(output, infos[i].fullname_);
                }

                output << "\n";
            }
            else if (format_ == "csv-short")
            {
                // first print raw value counters
                bool first = true;
                for (std::size_t i = 0; i != counter_shortnames_.size(); ++i)
                {
                    if (infos[i].type_ != performance_counters::counter_raw)
                        continue;
                    if (!first)
                        output << ",";
                    first = false;
                    print_name_csv_short(output, counter_shortnames_[i]);
                }

                // now print array value counters
                for (std::size_t i = 0; i != counter_shortnames_.size(); ++i)
                {
                    if (infos[i].type_ != performance_counters::counter_histogram)
                        continue;
                    if (!first)
                        output << ",";
                    first = false;
                    print_name_csv_short(output, counter_shortnames_[i]);
                }

                output << "\n";
            }
            csv_header_ = false;
        }
    }

    template <typename Stream, typename Value>
    void query_counters::print_values(Stream* output,
        std::vector<Value> && values, std::vector<std::size_t> && indicies,
        std::vector<performance_counters::counter_info> const& infos)
    {
        if (format_ == "csv" || format_ == "csv-short")
        {
            bool first = true;
            for (std::size_t i = 0; i != values.size(); ++i)
            {
                if (!first && output != nullptr)
                    *output << ",";
                first = false;
                print_value_csv(output, infos[i].fullname_, values[i]);
            }
            if (output != nullptr)
                *output << "\n";
        }
        else
        {
            std::size_t idx = 0;
            for (std::size_t i : indicies)
            {
                print_value(output, infos[i].fullname_, values[idx],
                    infos[i].unit_of_measure_);
                ++idx;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    bool query_counters::evaluate()
    {
        bool reset = false;
        if (get_config_entry("hpx.print_counter.reset", "0") == "1")
            reset = true;

        return evaluate_counters(reset);
    }

    void query_counters::terminate()
    {
        counters_.release();
    }

    ///////////////////////////////////////////////////////////////////////////
    void query_counters::start_counters(error_code& ec)
    {
        if (counters_.size() == 0)
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "query_counters::start_counters",
                "The counters to be evaluated have not been initialized yet");
            return;
        }

        // Start the performance counters.
        counters_.start(launch::sync, ec);
    }

    void query_counters::stop_counters(error_code& ec)
    {
        if (counters_.size() == 0)
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "query_counters::stop_counters",
                "The counters to be evaluated have not been initialized yet");
            return;
        }

        // Stop the performance counters.
        counters_.stop(launch::sync, ec);
    }

    void query_counters::reset_counters(error_code& ec)
    {
        if (counters_.size() == 0)
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "query_counters::reset_counters",
                "The counters to be evaluated have not been initialized yet");
            return;
        }

        // Reset the performance counters.
        counters_.reset(launch::sync, ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    bool query_counters::print_raw_counters(bool destination_is_cout,
        bool no_output, bool reset, char const* description,
        std::vector<performance_counters::counter_info> const& infos,
        error_code& ec)
    {
        // Query the performance counters.
        std::vector<std::size_t> indicies;
        indicies.reserve(infos.size());

        for (std::size_t i = 0; i != infos.size(); ++i)
        {
            if (infos[i].type_ != performance_counters::counter_raw)
                continue;
            indicies.push_back(i);
        }

        if (indicies.empty())
            return false;

        std::ostringstream output;
        if (description && !no_output)
            output << description << std::endl;

        std::vector<performance_counters::counter_value> values =
             counters_.get_counter_values(launch::sync, reset, ec);

        HPX_ASSERT(values.size() == indicies.size());

        // Output the performance counter value.
        if (!no_output)
            print_headers(output, infos);
        print_values(no_output ? nullptr : &output, std::move(values),
            std::move(indicies), infos);

        if (!no_output)
        {
            if (destination_is_cout)
            {
                std::cout << output.str() << std::flush;
            }
            else
            {
                std::ofstream out(destination_.c_str(), std::ofstream::app);
                out << output.str();
            }
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool query_counters::print_array_counters(bool destination_is_cout,
        bool no_output, bool reset, char const* description,
        std::vector<performance_counters::counter_info> const& infos,
        error_code& ec)
    {
        // Query the performance counters.
        std::vector<std::size_t> indicies;
        indicies.reserve(infos.size());

        for (std::size_t i = 0; i != infos.size(); ++i)
        {
            if (infos[i].type_ != performance_counters::counter_histogram)
                continue;
            indicies.push_back(i);
        }

        if (indicies.empty())
            return false;

        std::ostringstream output;
        if (description && !no_output)
            output << description << std::endl;

        std::vector<performance_counters::counter_values_array> values =
             counters_.get_counter_values_array(launch::sync, reset, ec);

        HPX_ASSERT(values.size() == indicies.size());

        // Output the performance counter value.
        if (!no_output)
            print_headers(output, infos);
        print_values(no_output ? nullptr : &output, std::move(values),
            std::move(indicies), infos);

        if (!no_output)
        {
            if (destination_is_cout)
            {
                std::cout << output.str() << std::flush;
            }
            else
            {
                std::ofstream out(destination_.c_str(), std::ofstream::app);
                out << output.str();
            }
        }
        return true;
    }

    bool query_counters::evaluate_counters(bool reset,
        char const* description, error_code& ec)
    {
        if (timer_.is_terminated())
        {
            // just do nothing as we're about to terminate the application
            counters_.release();       // free all held performance counters
            return false;
        }

        bool destination_is_cout = false;
        bool no_output = false;

        {
            std::lock_guard<mutex_type> l(mtx_);
            destination_is_cout = destination_ == "cout";
            no_output = destination_ == "none";
        }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
        // don't generate any console-output if the ITTNotify API is used
        if (!no_output && destination_is_cout && use_ittnotify_api)
            no_output = true;
#endif

        if (counters_.size() == 0)
        {
            // start has not been called yet
            HPX_THROWS_IF(ec, invalid_status,
                "query_counters::evaluate",
                "The counters to be evaluated have not been initialized yet");
            return false;
        }

        bool result = false;
        std::vector<performance_counters::counter_info> infos =
            counters_.get_counter_infos();

        result = print_raw_counters(destination_is_cout, no_output, reset,
            description, infos, ec);
        if (ec) return false;

        result = print_array_counters(destination_is_cout, no_output, reset,
            description, infos, ec) || result;
        if (ec) return false;

        if (&ec != &throws)
            ec = make_success_code();

        return result;
    }
}}

