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

#include <boost/assert.hpp>
#include <boost/foreach.hpp>

namespace hpx { namespace util
{
    query_counters::query_counters(std::vector<std::string> const& names,
            std::size_t interval, std::string const& dest)
      : names_(names), destination_(dest),
        timer_(boost::bind(&query_counters::evaluate, this),
            interval*1000, "query_counters", true)
    {
        // add counter prefix, if necessary
        BOOST_FOREACH(std::string& name, names_)
            performance_counters::ensure_counter_prefix(name);
    }

    void query_counters::find_counters()
    {
        mutex_type::scoped_lock l(mtx_);

        // do INI expansion on all counter names
        for (std::size_t i = 0; i < names_.size(); ++i)
            util::expand(names_[i]);

        if (ids_.empty())
        {
            ids_.reserve(names_.size());
            BOOST_FOREACH(std::string const& name, names_)
            {
                naming::id_type id;

                for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
                {
                    error_code ec;
                    id = performance_counters::get_counter(name, ec);
                    if (!ec) break;

                    using boost::posix_time::milliseconds;
                    threads::suspend(milliseconds(HPX_NETWORK_RETRIES_SLEEP));
                }

                if (HPX_UNLIKELY(!id))
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "query_counters::find_counters",
                        boost::str(boost::format(
                            "unknown performance counter: '%1%'") % name))
                }

                ids_.push_back(id);
            }
        }
        BOOST_ASSERT(ids_.size() == names_.size());
    }

    void query_counters::start()
    {
        // this will invoke the evaluate function for the first time
        timer_.start();
    }

    template <typename Stream>
    void print_value(Stream& out, std::string const& name,
        performance_counters::counter_value& value)
    {
        error_code ec;        // do not throw
        double val = value.get_value<double>(ec);

        out << name << ",";
        if (!ec)
            out << value.time_ << "," << val << "\n";
        else
            out << "invalid\n";
    }

    void query_counters::evaluate()
    {
        find_counters();

        for (std::size_t i = 0; i < names_.size(); ++i)
        {
            // Query the performance counter.
            using performance_counters::stubs::performance_counter;
            performance_counters::counter_value value =
                performance_counter::get_value(ids_[i]);

            // Output the performance counter value.
            mutex_type::scoped_lock l(mtx_);

            if (destination_ == "cout") {
                print_value(std::cout, names_[i], value);
            }
            else {
                std::ofstream out(destination_, std::ios_base::ate);
                print_value(out, names_[i], value);
            }
        }
    }
}}

