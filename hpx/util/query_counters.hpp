//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_QUERY_COUNTERS_SEP_27_2011_0255PM)
#define HPX_UTIL_QUERY_COUNTERS_SEP_27_2011_0255PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/include/performance_counters.hpp>

#include <string>
#include <vector>

#include <boost/cstdint.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT query_counters
    {
    public:
        query_counters(std::vector<std::string> const& names,
          std::size_t interval, std::string const& dest);

        void start();
        void evaluate();

    protected:
        void find_counters();

        template <typename Stream>
        void print_value(Stream& out, std::string const& name,
            performance_counters::counter_value& value);

    private:
        typedef lcos::local::mutex mutex_type;

        mutex_type mtx_;

        std::vector<std::string> names_;
        std::vector<naming::id_type> ids_;
        std::string destination_;

        interval_timer timer_;
        boost::uint64_t started_at_;
    };
}}

#endif
