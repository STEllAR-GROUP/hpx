//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCEL_COALESCING)
#include <hpx/functional/function.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/runtime/components/component_startup_shutdown.hpp>
#include <hpx/runtime_local/startup_function.hpp>
#include <hpx/string_util/classification.hpp>
#include <hpx/string_util/split.hpp>
#include <hpx/util/from_string.hpp>

#include <hpx/plugins/parcel/coalescing_counter_registry.hpp>

#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace hpx { namespace plugins { namespace parcel
{
    ///////////////////////////////////////////////////////////////////////////
    // Discoverer for the explicit (hand-rolled performance counter. The
    // purpose of this function is to invoke the supplied function f for all
    // allowed counter instance names supported by the counter type this
    // function has been registered with.
    bool counter_discoverer(
        hpx::performance_counters::counter_info const& info,
        hpx::performance_counters::discover_counter_func const& f,
        hpx::performance_counters::discover_counters_mode mode,
        hpx::error_code& ec)
    {
        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        bool result = coalescing_counter_registry::instance().
            counter_discoverer(info, p, f, mode, ec);
        if (!result || ec) return false;

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Creation function for explicit sine performance counter. It's purpose is
    // to create and register a new instance of the given name (or reuse an
    // existing instance).
    struct num_parcels_counter_surrogate
    {
        explicit num_parcels_counter_surrogate(std::string const& parameters)
          : parameters_(parameters)
        {}

        std::int64_t operator()(bool reset)
        {
            if (counter_.empty())
            {
                counter_ = coalescing_counter_registry::instance().
                    get_parcels_counter(parameters_);
                if (counter_.empty())
                    return 0;           // no counter available yet
            }

            // dispatch to actual counter
            return counter_(reset);
        }

        hpx::util::function_nonser<std::int64_t(bool)> counter_;
        std::string parameters_;
    };

    hpx::naming::gid_type num_parcels_counter_creator(
        hpx::performance_counters::counter_info const& info, hpx::error_code& ec)
    {
        switch (info.type_) {
        // NOLINTNEXTLINE(bugprone-branch-clone)
        case performance_counters::counter_monotonically_increasing:
            {
                performance_counters::counter_path_elements paths;
                performance_counters::get_counter_path_elements(
                    info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                if (paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "num_parcels_counter_creator",
                        "invalid counter name for number of parcels (instance "
                        "name must not be a valid base counter name)");
                    return naming::invalid_gid;
                }

                if (paths.parameters_.empty()) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "num_parcels_counter_creator",
                        "invalid counter parameter for number of parcels: must "
                        "specify an action type");
                    return naming::invalid_gid;
                }

                // ask registry
                hpx::util::function_nonser<std::int64_t(bool)> f =
                    coalescing_counter_registry::instance().
                        get_parcels_counter(paths.parameters_);

                if (!f.empty())
                {
                    return performance_counters::detail::create_raw_counter(
                        info, std::move(f), ec);
                }

                // the counter is not available yet, create surrogate function
                return performance_counters::detail::create_raw_counter(
                    info, num_parcels_counter_surrogate(paths.parameters_), ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter,
                "num_parcels_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct num_messages_counter_surrogate
    {
        explicit num_messages_counter_surrogate(std::string const& parameters)
          : parameters_(parameters)
        {}

        std::int64_t operator()(bool reset)
        {
            if (counter_.empty())
            {
                counter_ = coalescing_counter_registry::instance().
                    get_messages_counter(parameters_);
                if (counter_.empty())
                    return 0;           // no counter available yet
            }

            // dispatch to actual counter
            return counter_(reset);
        }

        hpx::util::function_nonser<std::int64_t(bool)> counter_;
        std::string parameters_;
    };

    hpx::naming::gid_type num_messages_counter_creator(
        hpx::performance_counters::counter_info const& info, hpx::error_code& ec)
    {
        switch (info.type_) {
        // NOLINTNEXTLINE(bugprone-branch-clone)
        case performance_counters::counter_monotonically_increasing:
            {
                performance_counters::counter_path_elements paths;
                performance_counters::get_counter_path_elements(
                    info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                if (paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "num_messages_counter_creator",
                        "invalid counter name for number of parcels (instance "
                        "name must not be a valid base counter name)");
                    return naming::invalid_gid;
                }

                if (paths.parameters_.empty()) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "num_messages_counter_creator",
                        "invalid counter parameter for number of parcels: must "
                        "specify an action type");
                    return naming::invalid_gid;
                }

                // ask registry
                hpx::util::function_nonser<std::int64_t(bool)> f =
                    coalescing_counter_registry::instance().
                        get_messages_counter(paths.parameters_);

                if (!f.empty())
                {
                    return performance_counters::detail::create_raw_counter(
                        info, std::move(f), ec);
                }

                // the counter is not available yet, create surrogate function
                return performance_counters::detail::create_raw_counter(
                    info, num_messages_counter_surrogate(paths.parameters_), ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter,
                "num_messages_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct num_parcels_per_message_counter_surrogate
    {
        explicit num_parcels_per_message_counter_surrogate(
            std::string const& parameters)
          : parameters_(parameters)
        {}

        std::int64_t operator()(bool reset)
        {
            if (counter_.empty())
            {
                counter_ = coalescing_counter_registry::instance().
                    get_parcels_per_message_counter(parameters_);
                if (counter_.empty())
                    return 0;           // no counter available yet
            }

            // dispatch to actual counter
            return counter_(reset);
        }

        hpx::util::function_nonser<std::int64_t(bool)> counter_;
        std::string parameters_;
    };

    hpx::naming::gid_type num_parcels_per_message_counter_creator(
        hpx::performance_counters::counter_info const& info, hpx::error_code& ec)
    {
        switch (info.type_) {
        // NOLINTNEXTLINE(bugprone-branch-clone)
        case performance_counters::counter_average_count:
            {
                performance_counters::counter_path_elements paths;
                performance_counters::get_counter_path_elements(
                    info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                if (paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "num_parcels_per_message_counter_creator",
                        "invalid counter name for number of parcels (instance "
                        "name must not be a valid base counter name)");
                    return naming::invalid_gid;
                }

                if (paths.parameters_.empty()) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "num_parcels_per_message_counter_creator",
                        "invalid counter parameter for number of parcels: must "
                        "specify an action type");
                    return naming::invalid_gid;
                }

                // ask registry
                hpx::util::function_nonser<std::int64_t(bool)> f =
                    coalescing_counter_registry::instance().
                        get_parcels_per_message_counter(paths.parameters_);

                if (!f.empty())
                {
                    return performance_counters::detail::create_raw_counter(
                        info, std::move(f), ec);
                }

                // the counter is not available yet, create surrogate function
                return performance_counters::detail::create_raw_counter(
                    info, num_parcels_per_message_counter_surrogate(
                        paths.parameters_), ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter,
                "num_parcels_per_message_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct average_time_between_parcels_counter_surrogate
    {
        explicit average_time_between_parcels_counter_surrogate(
                std::string const& parameters)
          : parameters_(parameters)
        {}

        std::int64_t operator()(bool reset)
        {
            if (counter_.empty())
            {
                counter_ = coalescing_counter_registry::instance().
                    get_average_time_between_parcels_counter(parameters_);
                if (counter_.empty())
                    return 0;           // no counter available yet
            }

            // dispatch to actual counter
            return counter_(reset);
        }

        hpx::util::function_nonser<std::int64_t(bool)> counter_;
        std::string parameters_;
    };

    hpx::naming::gid_type average_time_between_parcels_counter_creator(
        hpx::performance_counters::counter_info const& info, hpx::error_code& ec)
    {
        switch (info.type_) {
        case performance_counters::counter_average_timer:
            {
                performance_counters::counter_path_elements paths;
                performance_counters::get_counter_path_elements(
                    info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                if (paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "average_time_between_parcels_counter_creator",
                        "invalid counter name for number of parcels (instance "
                        "name must not be a valid base counter name)");
                    return naming::invalid_gid;
                }

                if (paths.parameters_.empty()) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "average_time_between_parcels_counter_creator",
                        "invalid counter parameter for number of parcels: must "
                        "specify an action type");
                    return naming::invalid_gid;
                }

                // ask registry
                hpx::util::function_nonser<std::int64_t(bool)> f =
                    coalescing_counter_registry::instance().
                        get_average_time_between_parcels_counter(
                            paths.parameters_);

                if (!f.empty())
                {
                    return performance_counters::detail::create_raw_counter(
                        info, std::move(f), ec);
                }

                // the counter is not available yet, create surrogate function
                return performance_counters::detail::create_raw_counter(info,
                    average_time_between_parcels_counter_surrogate(
                        paths.parameters_), ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter,
                "average_time_between_parcels_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    struct time_between_parcels_histogram_counter_surrogate
    {
        time_between_parcels_histogram_counter_surrogate(
                std::string const& action_name, std::int64_t min_boundary,
            std::int64_t max_boundary, std::int64_t num_buckets)
          : action_name_(action_name), min_boundary_(min_boundary),
            max_boundary_(max_boundary), num_buckets_(num_buckets)
        {}

        time_between_parcels_histogram_counter_surrogate(
                time_between_parcels_histogram_counter_surrogate const& rhs)
          : action_name_(rhs.action_name_), min_boundary_(rhs.min_boundary_),
            max_boundary_(rhs.max_boundary_), num_buckets_(rhs.num_buckets_)
        {}

        std::vector<std::int64_t> operator()(bool reset)
        {
            {
                std::lock_guard<hpx::lcos::local::spinlock> l(mtx_);
                if (counter_.empty())
                {
                    counter_ = coalescing_counter_registry::instance().
                        get_time_between_parcels_histogram_counter(action_name_,
                            min_boundary_, max_boundary_, num_buckets_);

                    // no counter available yet
                    if (counter_.empty())
                        return coalescing_counter_registry::empty_histogram(reset);
                }
            }

            // dispatch to actual counter
            return counter_(reset);
        }

        hpx::lcos::local::spinlock mtx_;
        hpx::util::function_nonser<std::vector<std::int64_t>(bool)> counter_;
        std::string action_name_;
        std::int64_t min_boundary_;
        std::int64_t max_boundary_;
        std::int64_t num_buckets_;
    };

    hpx::naming::gid_type time_between_parcels_histogram_counter_creator(
        hpx::performance_counters::counter_info const& info, hpx::error_code& ec)
    {
        switch (info.type_) {
        case performance_counters::counter_histogram:
            {
                performance_counters::counter_path_elements paths;
                performance_counters::get_counter_path_elements(
                    info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                if (paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "time_between_parcels_histogram_counter_creator",
                        "invalid counter name for "
                        "time-between-parcels histogram (instance "
                        "name must not be a valid base counter name)");
                    return naming::invalid_gid;
                }

                if (paths.parameters_.empty())
                {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "time_between_parcels_histogram_counter_creator",
                        "invalid counter parameter for "
                        "time-between-parcels histogram: must "
                        "specify an action type");
                    return naming::invalid_gid;
                }

                // split parameters, extract separate values
                std::vector<std::string> params;
                hpx::string_util::split(params, paths.parameters_,
                    hpx::string_util::is_any_of(","),
                    hpx::string_util::token_compress_mode::off);

                std::int64_t min_boundary = 0;
                std::int64_t max_boundary = 1000000;  // 1ms
                std::int64_t num_buckets = 20;

                if (params.empty() || params[0].empty())
                {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "time_between_parcels_histogram_counter_creator",
                        "invalid counter parameter for "
                        "time-between-parcels histogram: "
                        "must specify an action type");
                    return naming::invalid_gid;
                }

                if (params.size() > 1 && !params[1].empty())
                    min_boundary = util::from_string<std::int64_t>(params[1]);
                if (params.size() > 2 && !params[2].empty())
                    max_boundary = util::from_string<std::int64_t>(params[2]);
                if (params.size() > 3 && !params[3].empty())
                    num_buckets = util::from_string<std::int64_t>(params[3]);

                // ask registry
                hpx::util::function_nonser<std::vector<std::int64_t>(bool)> f =
                    coalescing_counter_registry::instance().
                        get_time_between_parcels_histogram_counter(params[0],
                            min_boundary, max_boundary, num_buckets);

                if (!f.empty())
                {
                    return performance_counters::detail::create_raw_counter(
                        info, std::move(f), ec);
                }

                // the counter is not available yet, create surrogate function
                return performance_counters::detail::create_raw_counter(info,
                    time_between_parcels_histogram_counter_surrogate(
                        params[0], min_boundary, max_boundary, num_buckets), ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter,
                "time_between_parcels_histogram_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // This function will be registered as a startup function for HPX below.
    //
    // That means it will be executed in a HPX-thread before hpx_main, but after
    // the runtime has been initialized and started.
    void startup()
    {
        using namespace hpx::performance_counters;

        // define the counter types
        generic_counter_type_data const counter_types[] =
        {
            // /coalescing(locality#<locality_id>/total)/count/parcels@action-name
            { "/coalescing/count/parcels", counter_monotonically_increasing,
              "returns the number of parcels handled by the message handler "
              "associated with the action which is given by the counter "
              "parameter",
              HPX_PERFORMANCE_COUNTER_V1,
              &num_parcels_counter_creator,
              &counter_discoverer,
              ""
            },
            // /coalescing(locality#<locality_id>/total)/count/messages@action-name
            { "/coalescing/count/messages", counter_monotonically_increasing,
              "returns the number of messages creates as the result of "
              "coalescing parcels of the action which is given by the counter "
              "parameter",
              HPX_PERFORMANCE_COUNTER_V1,
              &num_messages_counter_creator,
              &counter_discoverer,
              ""
            },
            // /coalescing(...)/count/average-parcels-per-message@action-name
            { "/coalescing/count/average-parcels-per-message",
              counter_average_count,
              "returns the average number of parcels sent in a message "
              "generated by the message handler associated with the action "
              "which is given by the counter parameter",
              HPX_PERFORMANCE_COUNTER_V1,
              &num_parcels_per_message_counter_creator,
              &counter_discoverer,
              ""
            },
            // /coalescing(...)/time/between-parcels-average@action-name
            { "/coalescing/time/between-parcels-average", counter_average_timer,
              "returns the average time between parcels for the "
              "action which is given by the counter parameter",
              HPX_PERFORMANCE_COUNTER_V1,
              &average_time_between_parcels_counter_creator,
              &counter_discoverer,
              "ns"
            },
            // /coalescing(...)/time/between-parcels-histogram@action-name,min,max,buckets
            { "/coalescing/time/between-parcels-histogram", counter_histogram,
              "returns the histogram for the times between parcels for "
              "the action which is given by the counter parameter",
              HPX_PERFORMANCE_COUNTER_V1,
              &time_between_parcels_histogram_counter_creator,
              &counter_discoverer,
              "ns/0.1%"
            }
        };

        // Install the counter types, un-installation of the types is handled
        // automatically.
        install_counter_types(counter_types,
            sizeof(counter_types)/sizeof(counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_startup(hpx::startup_function_type& startup_func,
        bool& pre_startup)
    {
        // return our startup-function if performance counters are required
        startup_func = startup;   // function to run during startup
        pre_startup = true;       // run 'startup' as pre-startup function
        return true;
    }
}}}

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a HPX-thread during
// runtime startup. We use this function to register our performance counter
// type and performance counter instances.
//
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_MODULE_DYNAMIC(hpx::plugins::parcel::get_startup);

#endif
