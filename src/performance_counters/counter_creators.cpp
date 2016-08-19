//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/agas/stubs/component_namespace.hpp>
#include <hpx/runtime/agas/stubs/primary_namespace.hpp>
#include <hpx/runtime/agas/stubs/symbol_namespace.hpp>
#include <hpx/runtime/agas/stubs/locality_namespace.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/util/function.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    // Creation functions to be registered with counter types

    /// Default discovery function for performance counters; to be registered
    /// with the counter types. It will pass the \a counter_info and the
    /// \a error_code to the supplied function.
    bool default_counter_discoverer(counter_info const& info,
        discover_counter_func const& f,
        discover_counters_mode mode, error_code& ec)
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
        discover_counter_func const& f,
        discover_counters_mode mode, error_code& ec)
    {
        performance_counters::counter_info i = info;

        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        if (mode == discover_counters_minimal ||
            p.parentinstancename_.empty() || p.instancename_.empty())
        {
            if (p.parentinstancename_.empty())
            {
                p.parentinstancename_ = "locality#*";
                p.parentinstanceindex_ = -1;
            }

            if (p.instancename_.empty())
            {
                p.instancename_ = "total";
                p.instanceindex_ = -1;
            }

            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;
        }
        else if (!f(i, ec) || ec) {
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }

    /// Default discoverer function for AGAS performance counters; to be
    /// registered with the counter types. It is suitable to be used for all
    /// counters following the naming scheme:
    ///
    ///   /<objectname>{locality#0/total}/<instancename>
    ///
    bool locality0_counter_discoverer(counter_info const& info,
        discover_counter_func const& f,
        discover_counters_mode mode, error_code& ec)
    {
        performance_counters::counter_info i = info;

        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        // restrict to locality zero
        if (p.parentinstancename_ == "locality#*")
        {
            p.parentinstancename_ = "locality";
            p.parentinstanceindex_ = 0;
        }

        if (mode == discover_counters_minimal ||
            p.parentinstancename_.empty() || p.instancename_.empty())
        {
            if (p.parentinstancename_.empty())
            {
                p.parentinstancename_ = "locality";
                p.parentinstanceindex_ = 0;
            }

            if (p.instancename_.empty())
            {
                p.instancename_ = "total";
                p.instanceindex_ = -1;
            }
        }

        status = get_counter_name(p, i.fullname_, ec);
        if (!status_is_valid(status) || !f(i, ec) || ec)
            return false;

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
        discover_counter_func const& f,
        discover_counters_mode mode, error_code& ec)
    {
        performance_counters::counter_info i = info;

        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        if (mode == discover_counters_minimal ||
            p.parentinstancename_.empty() || p.instancename_.empty())
        {
            if (p.parentinstancename_.empty())
            {
                p.parentinstancename_ = "locality#*";
                p.parentinstanceindex_ = -1;
            }

            if (p.instancename_.empty())
            {
                p.instancename_ = "total";
                p.instanceindex_ = -1;
            }

            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;

            p.instancename_ = "worker-thread#*";
            p.instanceindex_ = -1;

            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;
        }
        else if (!f(i, ec) || ec) {
            return false;
        }

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }

    /// Default discoverer function for performance counters; to be registered
    /// with the counter types. It is suitable to be used for all counters
    /// following the naming scheme:
    ///
    ///   /<objectname>(locality#<locality_id>/numa#<threadnum>)/<instancename>
    ///
    bool locality_numa_counter_discoverer(counter_info const& info,
        discover_counter_func const& f,
        discover_counters_mode mode, error_code& ec)
    {
        performance_counters::counter_info i = info;

        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        if (mode == discover_counters_minimal ||
            p.parentinstancename_.empty() || p.instancename_.empty())
        {
            if (p.parentinstancename_.empty())
            {
                p.parentinstancename_ = "locality#*";
                p.parentinstanceindex_ = -1;
            }

            if (p.instancename_.empty())
            {
                p.instancename_ = "total";
                p.instanceindex_ = -1;
            }

            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;

            p.instancename_ = "numa-node#*";
            p.instanceindex_ = -1;

            status = get_counter_name(p, i.fullname_, ec);
            if (!status_is_valid(status) || !f(i, ec) || ec)
                return false;
        }
        else if (!f(i, ec) || ec) {
            return false;
        }

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
        hpx::util::function_nonser<boost::int64_t(bool)> const& f, error_code& ec)
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

    namespace detail
    {
        naming::gid_type retrieve_agas_counter(std::string const& name,
            naming::id_type const& agas_id, error_code& ec)
        {
            naming::gid_type id;

            //  get action code from counter type
            agas::namespace_action_code service_code =
                agas::detail::retrieve_action_service_code(name, ec);
            if (agas::invalid_request == service_code) return id;

            // compose request
            agas::request req(service_code, name);
            agas::response rep;

            switch (service_code) {
            case agas::component_ns_statistics_counter:
                rep = agas::stubs::component_namespace::service(
                    agas_id, req, threads::thread_priority_default, ec);
                break;
            case agas::primary_ns_statistics_counter:
                rep = agas::stubs::primary_namespace::service(
                    agas_id, req, threads::thread_priority_default, ec);
                break;
            case agas::symbol_ns_statistics_counter:
                rep = agas::stubs::symbol_namespace::service(
                    agas_id, req, threads::thread_priority_default, ec);
                break;
            case agas::locality_ns_statistics_counter:
                rep = agas::stubs::locality_namespace::service(
                    agas_id, req, threads::thread_priority_default, ec);
                break;
            default:
                HPX_THROWS_IF(ec, bad_parameter, "retrieve_statistics_counter",
                    "unknown counter agas counter name: " + name);
                break;
            }
            if (!ec)
                id = rep.get_statistics_counter();
            return id;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Creation function for raw AGAS counters. This function checks the
    /// validity of the supplied counter name, it has to follow the scheme:
    ///
    ///   /agas(<objectinstance>/total)/<instancename>
    ///
    naming::gid_type agas_raw_counter_creator(
        counter_info const& info, error_code& ec, char const* const service_name)
    {
        // verify the validity of the counter instance name
        counter_path_elements paths;
        get_counter_path_elements(info.fullname_, paths, ec);
        if (ec) return naming::invalid_gid;

        if (paths.objectname_ != "agas") {
            HPX_THROWS_IF(ec, bad_parameter, "agas_raw_counter_creator",
                "unknown performance counter (unrelated to AGAS)");
            return naming::invalid_gid;
        }
        if (paths.parentinstance_is_basename_) {
            HPX_THROWS_IF(ec, bad_parameter, "agas_raw_counter_creator",
                "invalid counter instance parent name: " +
                    paths.parentinstancename_);
            return naming::invalid_gid;
        }

        // counter instance name: <agas_instance_name>/total
        // for instance: locality#0/total
        if (paths.instancename_ == "total" && paths.instanceindex_ == -1)
        {
            // find the referenced AGAS instance and dispatch the request there
            std::string service(agas::service_name);
            service += paths.parentinstancename_;

            if (-1 == paths.parentinstanceindex_) {
                HPX_THROWS_IF(ec, bad_parameter, "agas_raw_counter_creator",
                    "invalid parent instance index: -1");
                return naming::invalid_gid;
            }
            service += "#";
            service += std::to_string(paths.parentinstanceindex_);

            service += "/";
            service += service_name;

            naming::id_type id;
            bool result = agas::resolve_name(launch::sync, service, id, ec);
            if (!result) {
                HPX_THROWS_IF(ec, not_implemented,
                    "agas_raw_counter_creator",
                    "invalid counter name: " +
                        remove_counter_prefix(info.fullname_));
                return naming::invalid_gid;
            }

            return detail::retrieve_agas_counter(info.fullname_, id, ec);
        }

        HPX_THROWS_IF(ec, not_implemented, "agas_raw_counter_creator",
            "invalid counter type name: " + paths.instancename_);
        return naming::invalid_gid;
    }
}}
