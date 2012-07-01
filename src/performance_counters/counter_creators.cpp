//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/exception.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    // Creation functions to be registered with counter types

    /// Default discovery function for performance counters; to be registered
    /// with the counter types. It will pass the \a counter_info and the
    /// \a error_code to the supplied function.
    bool default_counter_discoverer(counter_info const& info,
        HPX_STD_FUNCTION<discover_counter_func> const& f, error_code& ec)
    {
        return f(info, ec);
    }

    /// Default discoverer function for performance counters; to be registered
    /// with the counter types. It is suitable to be used for all counters
    /// following the naming scheme:
    ///
    ///   /<objectname>{locality#<locality_id>/total}/<instancename>
    ///
    bool locality_counter_discoverer(counter_info const& info,
        HPX_STD_FUNCTION<discover_counter_func> const& f, error_code& ec)
    {
        performance_counters::counter_info i = info;

        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        p.parentinstancename_ = "locality#<*>";
        p.parentinstanceindex_ = -1;
        p.instancename_ = "total";
        p.instanceindex_ = -1;

        status = get_counter_name(p, i.fullname_, ec);
        if (!status_is_valid(status) || !f(i, ec) || ec)
            return false;

//         boost::uint32_t last_locality = get_num_localities();
//         for (boost::uint32_t l = 0; l < last_locality; ++l)
//         {
//             p.parentinstanceindex_ = static_cast<boost::int32_t>(l);
//             status = get_counter_name(p, i.fullname_, ec);
//             if (!status_is_valid(status) || !f(i, ec) || ec)
//                 return false;
//         }

        if (&ec != &throws)
            ec = make_success_code();
        return true;
    }

    /// Default discoverer function for performance counters; to be registered
    /// with the counter types. It is suitable to be used for all counters
    /// following the naming scheme:
    ///
    ///   /<objectname>{locality#<locality_id>/thread#<threadnum>}/<instancename>
    ///
    bool locality_thread_counter_discoverer(counter_info const& info,
        HPX_STD_FUNCTION<discover_counter_func> const& f, error_code& ec)
    {
        performance_counters::counter_info i = info;

        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        p.parentinstancename_ = "locality#<*>";
        p.parentinstanceindex_ = -1;
        p.instancename_ = "total";
        p.instanceindex_ = -1;

        status = get_counter_name(p, i.fullname_, ec);
        if (!status_is_valid(status) || !f(i, ec) || ec)
            return false;

        p.instancename_ = "worker-thread#<*>";
        p.instanceindex_ = -1;

        status = get_counter_name(p, i.fullname_, ec);
        if (!status_is_valid(status) || !f(i, ec) || ec)
            return false;

//         boost::uint32_t last_locality = get_num_localities();
//         std::size_t num_threads = get_os_thread_count();
//         for (boost::uint32_t l = 0; l <= last_locality; ++l)
//         {
//             p.parentinstanceindex_ = static_cast<boost::int32_t>(l);
//             p.instancename_ = "total";
//             p.instanceindex_ = -1;
//             status = get_counter_name(p, i.fullname_, ec);
//             if (!status_is_valid(status) || !f(i, ec) || ec)
//                 return false;
//
//             for (std::size_t t = 0; t < num_threads; ++t)
//             {
//                 p.instancename_ = "worker-thread";
//                 p.instanceindex_ = static_cast<boost::int32_t>(t);
//                 status = get_counter_name(p, i.fullname_, ec);
//                 if (!status_is_valid(status) || !f(i, ec) || ec)
//                     return false;
//             }
//         }

        if (&ec != &throws)
            ec = make_success_code();
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Creation function for raw counters. The passed function is encapsulating
    /// the actual value to monitor. This function checks the validity of the
    /// supplied counter name, it has to follow the scheme:
    ///
    ///   /<objectname>{locality#<locality_id>/total}/<instancename>
    ///
    naming::gid_type locality_raw_counter_creator(counter_info const& info,
        HPX_STD_FUNCTION<boost::int64_t()> const& f, error_code& ec)
    {
        // verify the validity of the counter instance name
        counter_path_elements paths;
        get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter, "locality_raw_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
            return detail::create_raw_counter(info, f, ec);   // overall counter

        HPX_THROWS_IF(ec, bad_parameter, "locality_raw_counter_creator",
            "invalid counter instance name: " + paths.instancename_);
        return naming::invalid_gid;
    }
}}
