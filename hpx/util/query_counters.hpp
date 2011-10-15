//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_QUERY_COUNTERS_SEP_27_2011_0255PM)
#define HPX_UTIL_QUERY_COUNTERS_SEP_27_2011_0255PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/interval_timer.hpp>
#include <hpx/lcos/local_mutex.hpp>
#include <hpx/include/performance_counters.hpp>

#include <string>
#include <vector>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT query_counters
    {
    public:

        query_counters(std::vector<std::string> const& names,
            std::size_t interval, std::ostream& out);

        void start();
        void evaluate();

    protected:
        void find_counters();

    private:
        typedef lcos::local_mutex mutex_type;

        mutex_type mtx_;
        std::ostream& out_;

        std::vector<std::string> names_;
        std::vector<naming::id_type> ids_;

        interval_timer timer_;
    };
}}

#endif
