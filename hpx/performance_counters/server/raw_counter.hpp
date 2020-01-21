//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PERFORMANCE_COUNTERS_SERVER_RAW_COUNTER_MAR_03_2009_0743M
#define HPX_PERFORMANCE_COUNTERS_SERVER_RAW_COUNTER_MAR_03_2009_0743M

#include <hpx/config.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/runtime/components/server/component_base.hpp>
#include <hpx/functional/function.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    class HPX_EXPORT raw_counter
      : public base_performance_counter,
        public components::component_base<raw_counter>
    {
        typedef components::component_base<raw_counter> base_type;

    public:
        typedef raw_counter type_holder;
        typedef base_performance_counter base_type_holder;

        raw_counter()
          : reset_(false)
        {}

        raw_counter(counter_info const& info,
            hpx::util::function_nonser<std::int64_t(bool)> f);

        hpx::performance_counters::counter_value
            get_counter_value(bool reset = false);
        void reset_counter_value();

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        void finalize()
        {
            base_performance_counter::finalize();
            base_type::finalize();
        }

    private:
        hpx::util::function_nonser<std::int64_t(bool)> f_;
        bool reset_;
    };
}}}

#endif
