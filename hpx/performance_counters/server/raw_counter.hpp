//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_SERVER_RAW_COUNTER_MAR_03_2009_0743M)
#define HPX_PERFORMANCE_COUNTERS_SERVER_RAW_COUNTER_MAR_03_2009_0743M

#include <hpx/hpx_fwd.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    class raw_counter 
      : public base_performance_counter
    {
    public:
        void get_counter_info(counter_info& info);
        void get_counter_value(counter_value& value);
    };

}}}

#endif
