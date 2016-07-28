//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_AGAS_PRIMARY_NAMESPACE_COUNTERS_JUN_29_2106_0153PM)
#define HPX_AGAS_PRIMARY_NAMESPACE_COUNTERS_JUN_29_2106_0153PM

#include <hpx/config.hpp>
#include <hpx/runtime/agas/detail/update_time_on_exit.hpp>
#include <hpx/runtime/agas/namespace_action_code.hpp>
#include <hpx/util/function.hpp>

#include <boost/atomic.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace agas { namespace server
{
    struct primary_namespace;
}}}

namespace hpx { namespace agas { namespace detail
{
    // data structure holding all counters for the omponent_namespace component
    struct primary_namespace_counter_data
    {
    private:
        HPX_NON_COPYABLE(primary_namespace_counter_data);

    public:
        struct api_counter_data
        {
            api_counter_data()
              : count_(0)
              , time_(0)
            {}

            boost::atomic<boost::int64_t> count_;
            boost::atomic<boost::int64_t> time_;
        };

        primary_namespace_counter_data() {}

    public:
        // access current counter values
        boost::int64_t get_route_count(bool);
        boost::int64_t get_bind_gid_count(bool);
        boost::int64_t get_resolve_gid_count(bool);
        boost::int64_t get_unbind_gid_count(bool);
        boost::int64_t get_increment_credit_count(bool);
        boost::int64_t get_decrement_credit_count(bool);
        boost::int64_t get_allocate_count(bool);
        boost::int64_t get_begin_migration_count(bool);
        boost::int64_t get_end_migration_count(bool);
        boost::int64_t get_overall_count(bool);

        boost::int64_t get_route_time(bool);
        boost::int64_t get_bind_gid_time(bool);
        boost::int64_t get_resolve_gid_time(bool);
        boost::int64_t get_unbind_gid_time(bool);
        boost::int64_t get_increment_credit_time(bool);
        boost::int64_t get_decrement_credit_time(bool);
        boost::int64_t get_allocate_time(bool);
        boost::int64_t get_begin_migration_time(bool);
        boost::int64_t get_end_migration_time(bool);
        boost::int64_t get_overall_time(bool);

        // increment counter values
        void increment_route_count();
        void increment_bind_gid_count();
        void increment_resolve_gid_count();
        void increment_unbind_gid_count();
        void increment_increment_credit_count();
        void increment_decrement_credit_count();
        void increment_allocate_count();
        void increment_begin_migration_count();
        void increment_end_migration_count();

        util::function_nonser<boost::int64_t(bool)>
        get_counter_function(
            counter_target target, namespace_action_code code, error_code& ec);

    private:
        friend struct update_time_on_exit;
        friend struct local_primary_namespace;
        friend struct agas::server::primary_namespace;

        api_counter_data route_;                // primary_ns_
        api_counter_data bind_gid_;             // primary_ns_bind_gid
        api_counter_data resolve_gid_;          // primary_ns_resolve_gid
        api_counter_data unbind_gid_;           // primary_ns_unbind_gid
        api_counter_data increment_credit_;     // primary_ns_increment_credit
        api_counter_data decrement_credit_;     // primary_ns_decrement_credit
        api_counter_data allocate_;             // primary_ns_allocate
        api_counter_data begin_migration_;      // primary_ns_begin_migration
        api_counter_data end_migration_;        // primary_ns_end_migration
    };
}}}

#endif
