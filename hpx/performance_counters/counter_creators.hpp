//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_CREATORS_FEB_12_2012_0401PM)
#define HPX_PERFORMANCE_COUNTERS_CREATORS_FEB_12_2012_0401PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/util/function.hpp>

#include <boost/cstdint.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    // Discoverer functions to be registered with counter types

    /// Default discovery function for performance counters; to be registered
    /// with the counter types. It will pass the \a counter_info and the
    /// \a error_code to the supplied function.
    HPX_API_EXPORT bool default_counter_discoverer(counter_info const&,
        discover_counter_func const&, discover_counters_mode, error_code&);

    /// Default discoverer function for performance counters; to be registered
    /// with the counter types. It is suitable to be used for all counters
    /// following the naming scheme:
    ///
    ///   /<objectname>(locality#<locality_id>/total)/<instancename>
    ///
    HPX_API_EXPORT bool locality_counter_discoverer(counter_info const&,
        discover_counter_func const&, discover_counters_mode, error_code&);

    /// Default discoverer function for AGAS performance counters; to be
    /// registered with the counter types. It is suitable to be used for all
    /// counters following the naming scheme:
    ///
    ///   /<objectname>{locality#0/total}/<instancename>
    ///
    HPX_API_EXPORT bool locality0_counter_discoverer(counter_info const&,
        discover_counter_func const&, discover_counters_mode, error_code&);

    /// Default discoverer function for performance counters; to be registered
    /// with the counter types. It is suitable to be used for all counters
    /// following the naming scheme:
    ///
    ///   /<objectname>(locality#<locality_id>/thread#<threadnum>)/<instancename>
    ///
    HPX_API_EXPORT bool locality_thread_counter_discoverer(counter_info const&,
        discover_counter_func const&, discover_counters_mode, error_code&);

    ///////////////////////////////////////////////////////////////////////////
    /// Creation function for raw counters. The passed function is encapsulating
    /// the actual value to monitor. This function checks the validity of the
    /// supplied counter name, it has to follow the scheme:
    ///
    ///   /<objectname>(locality#<locality_id>/total)/<instancename>
    ///
    HPX_API_EXPORT naming::gid_type locality_raw_counter_creator(
        counter_info const&, hpx::util::function_nonser<boost::int64_t(bool)> const&,
        error_code&);

    ///////////////////////////////////////////////////////////////////////////
    /// Creation function for raw counters. The passed function is encapsulating
    /// the actual value to monitor. This function checks the validity of the
    /// supplied counter name, it has to follow the scheme:
    ///
    ///   /agas(<objectinstance>/total)/<instancename>
    ///
    HPX_API_EXPORT naming::gid_type agas_raw_counter_creator(
        counter_info const&, error_code&, char const* const);

    /// Default discoverer function for performance counters; to be registered
    /// with the counter types. It is suitable to be used for all counters
    /// following the naming scheme:
    ///
    ///   /agas(<objectinstance>/total)/<instancename>
    ///
    HPX_API_EXPORT bool agas_counter_discoverer(counter_info const&,
        discover_counter_func const&, discover_counters_mode, error_code&);

    ///////////////////////////////////////////////////////////////////////////
    // Creation function for action invocation counters.
    HPX_API_EXPORT naming::gid_type action_invocation_counter_creator(
        counter_info const&, error_code&);

    // Discoverer function for action invocation counters.
    HPX_API_EXPORT bool action_invocation_counter_discoverer(counter_info const&,
        discover_counter_func const&, discover_counters_mode, error_code&);
}}

#endif

