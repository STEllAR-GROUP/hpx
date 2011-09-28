//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/query_counters.hpp>

namespace hpx { namespace util
{
    query_counters::query_counters(std::vector<std::string> const& names, 
            std::size_t interval, std::ostream& out)
      : names_(names), out_(out),
        timer_(boost::bind(&query_counters::evaluate, this),
            interval*1000, "query_counters")
    {}

    void query_counters::start() 
    {
        timer_.start();
    }

    void query_counters::evaluate()
    {
        BOOST_FOREACH(std::string const& name, names_)
        {
            error_code ec;
            naming::gid_type gid;
            naming::get_agas_client().queryid(name, gid, ec);

            if (HPX_UNLIKELY(ec || !gid))
            {
                HPX_THROW_EXCEPTION(bad_parameter, "query_counters",
                    boost::str(boost::format(
                        "unknown performance counter: '%s'") % name))
            }

            // Query the performance counter.
            using performance_counters::stubs::performance_counter;
            performance_counters::counter_value value = 
                performance_counter::get_value(gid);
            double val = value.get_value<double>(ec);
            if (!ec)
                out_ << name << "," << value.time_ << "," << val << "\n"; 
            else
                out_ << name << ",invalid\n"; 
        }
    }
}}

