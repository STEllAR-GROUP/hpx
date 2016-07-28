//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/agas/detail/primary_namespace_counter_data.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <boost/cstdint.hpp>

namespace hpx { namespace agas { namespace detail
{
    util::function_nonser<boost::int64_t(bool)>
    primary_namespace_counter_data::get_counter_function(
        counter_target target, namespace_action_code code, error_code& ec)
    {
        util::function_nonser<boost::int64_t(bool)> get_data_func;
        if (target == detail::counter_target_count)
        {
            switch (code) {
            case primary_ns_route:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_route_count,
                    this, util::placeholders::_1);
                break;
            case primary_ns_bind_gid:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_bind_gid_count,
                    this, util::placeholders::_1);
                break;
            case primary_ns_resolve_gid:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_resolve_gid_count,
                    this, util::placeholders::_1);
                break;
            case primary_ns_unbind_gid:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_unbind_gid_count,
                    this, util::placeholders::_1);
                break;
            case primary_ns_increment_credit:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_increment_credit_count,
                    this, util::placeholders::_1);
                break;
            case primary_ns_decrement_credit:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_decrement_credit_count,
                    this, util::placeholders::_1);
                break;
            case primary_ns_allocate:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_allocate_count,
                    this, util::placeholders::_1);
                break;
            case primary_ns_begin_migration:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_begin_migration_count,
                    this, util::placeholders::_1);
                break;
            case primary_ns_end_migration:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_end_migration_count,
                    this, util::placeholders::_1);
                break;
            case primary_ns_statistics_counter:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_overall_count,
                    this, util::placeholders::_1);
                break;
            default:
                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace_counter_data::get_counter_function"
                  , "bad action code while querying statistics");
                return get_data_func;
            }
        }
        else {
            HPX_ASSERT(detail::counter_target_time == target);
            switch (code) {
            case primary_ns_route:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_route_time,
                    this, util::placeholders::_1);
                break;
            case primary_ns_bind_gid:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_bind_gid_time,
                    this, util::placeholders::_1);
                break;
            case primary_ns_resolve_gid:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_resolve_gid_time,
                    this, util::placeholders::_1);
                break;
            case primary_ns_unbind_gid:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_unbind_gid_time,
                    this, util::placeholders::_1);
                break;
            case primary_ns_increment_credit:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_increment_credit_time,
                    this, util::placeholders::_1);
                break;
            case primary_ns_decrement_credit:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_decrement_credit_time,
                    this, util::placeholders::_1);
                break;
            case primary_ns_allocate:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_allocate_time,
                    this, util::placeholders::_1);
                break;
            case primary_ns_begin_migration:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_begin_migration_time,
                    this, util::placeholders::_1);
                break;
            case primary_ns_end_migration:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_end_migration_time,
                    this, util::placeholders::_1);
                break;
            case primary_ns_statistics_counter:
                get_data_func = util::bind(
                    &primary_namespace_counter_data::get_overall_time,
                    this, util::placeholders::_1);
                break;
            default:
                HPX_THROWS_IF(ec, bad_parameter
                  , "primary_namespace_counter_data::get_counter_function"
                  , "bad action code while querying statistics");
                return get_data_func;
            }
        }

        if (&ec != &throws)
            ec = make_success_code();

        return get_data_func;
    }

    // access current counter values
    boost::int64_t primary_namespace_counter_data::get_route_count(bool reset)
    {
        return util::get_and_reset_value(route_.count_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_bind_gid_count(
        bool reset)
    {
        return util::get_and_reset_value(bind_gid_.count_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_resolve_gid_count(
        bool reset)
    {
        return util::get_and_reset_value(resolve_gid_.count_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_unbind_gid_count(
        bool reset)
    {
        return util::get_and_reset_value(unbind_gid_.count_, reset);
    }

    boost::int64_t
    primary_namespace_counter_data::get_increment_credit_count(bool reset)
    {
        return util::get_and_reset_value(increment_credit_.count_, reset);
    }

    boost::int64_t
    primary_namespace_counter_data::get_decrement_credit_count(bool reset)
    {
        return util::get_and_reset_value(decrement_credit_.count_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_allocate_count(
        bool reset)
    {
        return util::get_and_reset_value(allocate_.count_, reset);
    }

    boost::int64_t
    primary_namespace_counter_data::get_begin_migration_count(bool reset)
    {
        return util::get_and_reset_value(begin_migration_.count_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_end_migration_count(
        bool reset)
    {
        return util::get_and_reset_value(end_migration_.count_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_overall_count(
        bool reset)
    {
        return util::get_and_reset_value(route_.count_, reset) +
            util::get_and_reset_value(bind_gid_.count_, reset) +
            util::get_and_reset_value(resolve_gid_.count_, reset) +
            util::get_and_reset_value(unbind_gid_.count_, reset) +
            util::get_and_reset_value(increment_credit_.count_, reset) +
            util::get_and_reset_value(decrement_credit_.count_, reset) +
            util::get_and_reset_value(allocate_.count_, reset) +
            util::get_and_reset_value(begin_migration_.count_, reset) +
            util::get_and_reset_value(end_migration_.count_, reset);
    }

    // access execution time counters
    boost::int64_t primary_namespace_counter_data::get_route_time(
        bool reset)
    {
        return util::get_and_reset_value(route_.time_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_bind_gid_time(
        bool reset)
    {
        return util::get_and_reset_value(bind_gid_.time_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_resolve_gid_time(
        bool reset)
    {
        return util::get_and_reset_value(resolve_gid_.time_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_unbind_gid_time(
        bool reset)
    {
        return util::get_and_reset_value(unbind_gid_.time_, reset);
    }

    boost::int64_t
    primary_namespace_counter_data::get_increment_credit_time(bool reset)
    {
        return util::get_and_reset_value(increment_credit_.time_, reset);
    }

    boost::int64_t
    primary_namespace_counter_data::get_decrement_credit_time(bool reset)
    {
        return util::get_and_reset_value(decrement_credit_.time_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_allocate_time(
        bool reset)
    {
        return util::get_and_reset_value(allocate_.time_, reset);
    }

    boost::int64_t
    primary_namespace_counter_data::get_begin_migration_time(bool reset)
    {
        return util::get_and_reset_value(begin_migration_.time_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_end_migration_time(
        bool reset)
    {
        return util::get_and_reset_value(end_migration_.time_, reset);
    }

    boost::int64_t primary_namespace_counter_data::get_overall_time(
        bool reset)
    {
        return util::get_and_reset_value(route_.time_, reset) +
            util::get_and_reset_value(bind_gid_.time_, reset) +
            util::get_and_reset_value(resolve_gid_.time_, reset) +
            util::get_and_reset_value(unbind_gid_.time_, reset) +
            util::get_and_reset_value(increment_credit_.time_, reset) +
            util::get_and_reset_value(decrement_credit_.time_, reset) +
            util::get_and_reset_value(allocate_.time_, reset) +
            util::get_and_reset_value(begin_migration_.time_, reset) +
            util::get_and_reset_value(end_migration_.time_, reset);
    }

    // increment counter values
    void primary_namespace_counter_data::increment_route_count()
    {
        ++route_.count_;
    }

    void primary_namespace_counter_data::increment_bind_gid_count()
    {
        ++bind_gid_.count_;
    }

    void primary_namespace_counter_data::increment_resolve_gid_count()
    {
        ++resolve_gid_.count_;
    }

    void primary_namespace_counter_data::increment_unbind_gid_count()
    {
        ++unbind_gid_.count_;
    }

    void primary_namespace_counter_data::increment_increment_credit_count()
    {
        ++increment_credit_.count_;
    }

    void primary_namespace_counter_data::increment_decrement_credit_count()
    {
        ++decrement_credit_.count_;
    }

    void primary_namespace_counter_data::increment_allocate_count()
    {
        ++allocate_.count_;
    }

    void primary_namespace_counter_data::increment_begin_migration_count()
    {
        ++begin_migration_.count_;
    }

    void primary_namespace_counter_data::increment_end_migration_count()
    {
        ++end_migration_.count_;
    }
}}}
