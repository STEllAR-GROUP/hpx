//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/util/query_counters.hpp>

namespace hpx { namespace util
{
    query_counters::query_counters(std::vector<std::string> const& names, 
            std::size_t interval, std::ostream& out)
      : out_(out), names_(names), 
        timer_(boost::bind(&query_counters::evaluate, this),
            interval*1000, "query_counters", true)
    {}

    void query_counters::start() 
    {
        ids_.reserve(names_.size());
        BOOST_FOREACH(std::string const& name, names_)
        {
            error_code ec;
            ids_.push_back(naming::invalid_id);
            agas::query_name(name, ids_.back(), ec);

            if (HPX_UNLIKELY(ec || !ids_.back()))
            {
                HPX_THROW_EXCEPTION(bad_parameter, "query_counters",
                    boost::str(boost::format(
                        "unknown performance counter: '%s'") % name))
            }
        }

        // this will invoke the evaluate function for the first time
        timer_.start();
    }

    void query_counters::evaluate()
    {
        BOOST_ASSERT(ids_.size() == names_.size());

        for (std::size_t i = 0; i < names_.size(); ++i) 
        {
            error_code ec;

            // Query the performance counter.
            using performance_counters::stubs::performance_counter;
            performance_counters::counter_value value = 
                performance_counter::get_value(ids_[i]);
            double val = value.get_value<double>(ec);

            if (!ec)
                out_ << names_[i] << "," << value.time_ << "," << val << "\n"; 
            else
                out_ << names_[i] << ",invalid\n"; 
        }
    }
}}

