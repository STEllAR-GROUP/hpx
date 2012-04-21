//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/high_resolution_clock.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>

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
            interval*1000, "query_counters", true),
        started_at_(0)
    {
        // add counter prefix, if necessary
        BOOST_FOREACH(std::string& name, names_)
            performance_counters::ensure_counter_prefix(name);
    }

    void query_counters::find_counters()
    {
        mutex_type::scoped_lock l(mtx_);
        if (ids_.empty())
        {
            // do INI expansion on all counter names
            for (std::size_t i = 0; i < names_.size(); ++i)
                util::expand(names_[i]);

            ids_.reserve(names_.size());
            BOOST_FOREACH(std::string const& name, names_)
            {
                error_code ec;
                naming::id_type id =
                    performance_counters::get_counter(name, ec);
                if (HPX_UNLIKELY(!id))
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "query_counters::find_counters",
                        boost::str(boost::format(
                            "unknown performance counter: '%1%'") % name))
                }

                ids_.push_back(id);

                using performance_counters::stubs::performance_counter;
                performance_counters::counter_info info =
                    performance_counter::get_info(id);
                uoms_.push_back(info.unit_of_measure_);
            }
        }
        BOOST_ASSERT(ids_.size() == names_.size());
    }

    void query_counters::start()
    {
        find_counters();

        for (std::size_t i = 0; i < names_.size(); ++i)
        {
            // start the performance counter
            using performance_counters::stubs::performance_counter;
            performance_counter::start(ids_[i]);
        }

        // this will invoke the evaluate function for the first time
        started_at_ = hpx::performance_counters::high_resolution_clock::now();
        timer_.start();
    }

    template <typename Stream>
    void query_counters::print_value(Stream& out, std::string const& name,
      performance_counters::counter_value& value, std::string const& uom)
    {
        error_code ec;        // do not throw
        double val = value.get_value<double>(ec);

        out << name << ",";
        if (!ec) {
            double elapsed = static_cast<double>(value.time_ - started_at_) * 1e-9;
            out << boost::str(boost::format("%.6f") % elapsed)
                << "[s]," << val;
            if (!uom.empty())
                out << "[" << uom << "]";
            out << "\n";
        }
        else {
            out << "invalid\n";
        }
    }

    void query_counters::evaluate()
    {
        if (timer_.is_terminated())
        {
            // too late, the counter has already been terminated
            HPX_THROW_EXCEPTION(invalid_status,
                "query_counters::evaluate",
                "The counters to be evaluated have been already terminated");
            return;
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
            HPX_THROW_EXCEPTION(invalid_status,
                "query_counters::evaluate",
                "The counters to be evaluated have not been initialized yet");
            return;
        }

        for (std::size_t i = 0; i < names_.size(); ++i)
        {
            // Query the performance counter.
            using performance_counters::stubs::performance_counter;
            performance_counters::counter_value value =
                performance_counter::get_value(ids_[i]);

            // Output the performance counter value.
            if (destination_is_cout) {
                print_value(std::cout, names_[i], value, uoms_[i]);
            }
            else {
                std::ofstream out(destination_, std::ios_base::ate);
                print_value(out, names_[i], value, uoms_[i]);
            }
        }
    }
}}

