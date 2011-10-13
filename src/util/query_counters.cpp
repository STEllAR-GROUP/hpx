//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/query_counters.hpp>

namespace hpx { namespace util
{
    query_counters::query_counters(std::vector<std::string> const& names,
            std::size_t interval, std::ostream& out)
      : out_(out), names_(names),
        timer_(boost::bind(&query_counters::evaluate, this),
            interval*1000, "query_counters", true)
    {}

    void query_counters::find_counters_locked()
    {
        if (ids_.empty())
        {
            ids_.reserve(names_.size());
            BOOST_FOREACH(std::string const& name, names_)
            {
                ids_.push_back(naming::invalid_id);

                for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
                {
                    if (agas::resolve_name(name, ids_.back()))
                        break;

                    threads::suspend(
                        boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
                }

                if (HPX_UNLIKELY(!ids_.back()))
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "query_counters::find_counters_locked",
                        boost::str(boost::format(
                            "unknown performance counter: '%1%'") % name))
                }
            }
        }

        if (HPX_UNLIKELY(ids_.size() != names_.size()))
            HPX_THROW_EXCEPTION(bad_parameter,
                "query_counters::find_counters_locked",
                "couldn't find all target counters");
    }

    void query_counters::start()
    {
        if (ids_.empty())
        {
            mutex_type::scoped_lock l(ids_mtx_);
            find_counters_locked();
        }

        // this will invoke the evaluate function for the first time
        timer_.start();
    }

    void query_counters::evaluate()
    {
        if (ids_.empty())
        {
            mutex_type::scoped_lock l(ids_mtx_);
            find_counters_locked();
        }

        for (std::size_t i = 0; i < names_.size(); ++i)
        {
            error_code ec;

            // Query the performance counter.
            using performance_counters::stubs::performance_counter;
            performance_counters::counter_value value =
                performance_counter::get_value(ids_[i]);
            double val = value.get_value<double>(ec);

            {
                mutex_type::scoped_lock l(io_mtx_);

                if (!ec)
                    out_ << names_[i] << "," << value.time_ << "," << val << "\n";
                else
                    out_ << names_[i] << ",invalid\n";
            }
        }
    }
}}

