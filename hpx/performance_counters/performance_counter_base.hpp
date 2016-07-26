//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_PERFORMANCE_COUNTER_BASE_SEP_18_2014_0732PM)
#define HPX_PERFORMANCE_COUNTERS_PERFORMANCE_COUNTER_BASE_SEP_18_2014_0732PM

#include <hpx/config.hpp>
#include <hpx/performance_counters/counters.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    //[performance_counter_interface
    // Abstract base interface for all Performance Counters.
    struct performance_counter_base
    {
        //<-
        /// Destructor, needs to be virtual to allow for clean destruction of
        /// derived objects
        virtual ~performance_counter_base() {}
        //->
        // Retrieve the descriptive information about the Performance Counter.
        virtual counter_info get_counter_info() const = 0;

        // Retrieve the current Performance Counter value.
        virtual counter_value get_counter_value(bool reset = false) = 0;

        // Retrieve the current Performance Counter value.
        virtual counter_values_array get_counter_values_array(
            bool reset = false) = 0;

        // Reset the Performance Counter (value).
        virtual void reset_counter_value() = 0;

        // Set the (initial) value of the Performance Counter.
        virtual void set_counter_value(counter_value const& /*value*/) = 0;

        // Start the Performance Counter.
        virtual bool start() = 0;

        // Stop the Performance Counter.
        virtual bool stop() = 0;
    };
    //]
}}

#endif
