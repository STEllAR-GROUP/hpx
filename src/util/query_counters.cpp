//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/query_counters.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <boost/assert.hpp>
#include <boost/foreach.hpp>

namespace hpx { namespace util
{
    query_counters::query_counters(std::vector<std::string> const& names,
            std::size_t interval, std::ostream& out)
      : out_(out), names_(names),
        timer_(boost::bind(&query_counters::evaluate, this),
            interval*1000, "query_counters", true)
    {
        // add counter prefix, if necessary
        BOOST_FOREACH(std::string& name, names_)
        {
            if (0 != name.find(hpx::performance_counters::counter_prefix))
                name = hpx::performance_counters::counter_prefix + name;
        }
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
                ids_.push_back(naming::invalid_id);

                for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
                {
                    if (agas::resolve_name(name, ids_.back()))
                        break;

                    using boost::posix_time::milliseconds;
                    threads::suspend(milliseconds(HPX_NETWORK_RETRIES_SLEEP));
                }

                if (HPX_UNLIKELY(!ids_.back()))
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "query_counters::find_counters",
                        boost::str(boost::format(
                            "unknown performance counter: '%1%'") % name))
                }
            }
        }
        BOOST_ASSERT(ids_.size() == names_.size());
    }

    void query_counters::start()
    {
        // this will invoke the evaluate function for the first time
        timer_.start();
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

            error_code ec;        // do not throw
            double val = value.get_value<double>(ec);

            // Output the performance counter value.
            mutex_type::scoped_lock l(mtx_);

            out_ << names_[i] << ",";
            if (!ec)
                out_ << value.time_ << "," << val << "\n";
            else
                out_ << "invalid\n";
        }
    }
}}

