//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_PERFORMANCE_COUNTER_JAN_18_2013_0939AM)
#define HPX_PERFORMANCE_COUNTERS_PERFORMANCE_COUNTER_JAN_18_2013_0939AM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/launch_policy.hpp>

#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>

#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT performance_counter
      : components::client_base<performance_counter, stubs::performance_counter>
    {
        typedef components::client_base<
            performance_counter, stubs::performance_counter
        > base_type;

        performance_counter() {}

        performance_counter(std::string const& name);

        performance_counter(std::string const& name, hpx::id_type const& locality);

        performance_counter(future<id_type> && id)
          : base_type(std::move(id))
        {}

        performance_counter(hpx::future<performance_counter> && c)
          : base_type(std::move(c))
        {}

        ///////////////////////////////////////////////////////////////////////
        future<counter_info> get_info() const;
        counter_info get_info(launch::sync_policy,
            error_code& ec = throws) const;

        future<counter_value> get_counter_value(bool reset = false);
        counter_value get_counter_value(launch::sync_policy,
            bool reset = false, error_code& ec = throws);

        future<counter_value> get_counter_value() const;
        counter_value get_counter_value(launch::sync_policy,
            error_code& ec = throws) const;

        future<counter_values_array> get_counter_values_array(bool reset = false);
        counter_values_array get_counter_values_array(launch::sync_policy,
            bool reset = false, error_code& ec = throws);

        future<counter_values_array> get_counter_values_array() const;
        counter_values_array get_counter_values_array(launch::sync_policy,
            error_code& ec = throws) const;

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        counter_info get_info_sync(error_code& ec = throws) const
        {
            return get_info(launch::sync, ec);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        counter_value get_counter_value_sync(bool reset = false,
            error_code& ec = throws)
        {
            return get_counter_value(launch::sync, reset, ec);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        counter_value get_counter_value_sync(error_code& ec = throws) const
        {
            return get_counter_value(launch::sync, ec);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        counter_values_array get_counter_values_array_sync(bool reset = false,
            error_code& ec = throws)
        {
            return get_counter_values_array(launch::sync, reset, ec);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        counter_values_array get_counter_values_array_sync(
            error_code& ec = throws) const
        {
            return get_counter_values_array(launch::sync, ec);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        future<bool> start();
        bool start(launch::sync_policy, error_code& ec = throws);

        future<bool> stop();
        bool stop(launch::sync_policy, error_code& ec = throws);

        future<void> reset();
        void reset(launch::sync_policy, error_code& ec = throws);

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        bool start_sync(error_code& ec = throws)
        {
            return start(launch::sync, ec);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        bool stop_sync(error_code& ec = throws)
        {
            return stop(launch::sync, ec);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        void reset_sync(error_code& ec = throws)
        {
            return reset(launch::sync, ec);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        future<std::string> get_name() const;
        std::string get_name(launch::sync_policy, error_code& ec = throws) const;

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        std::string get_name_sync(error_code& ec = throws) const
        {
            return get_name(launch::sync, ec);
        }
#endif

    private:
        template <typename T>
        static T extract_value(future<counter_value> && value)
        {
            return value.get().get_value<T>();
        }

    public:
        template <typename T>
        future<T> get_value(bool reset = false)
        {
            return get_counter_value(reset).then(
                util::bind(
                    &performance_counter::extract_value<T>,
                    util::placeholders::_1));
        }
        template <typename T>
        T get_value(launch::sync_policy, bool reset = false,
            error_code& ec = throws)
        {
            return get_counter_value(launch::sync, reset).get_value<T>(ec);
        }

        template <typename T>
        future<T> get_value() const
        {
            return get_counter_value().then(
                util::bind(
                    &performance_counter::extract_value<T>,
                    util::placeholders::_1));
        }
        template <typename T>
        T get_value(launch::sync_policy, error_code& ec = throws) const
        {
            return get_counter_value(launch::sync).get_value<T>(ec);
        }

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        template <typename T>
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        T get_value_sync(bool reset = false, error_code& ec = throws)
        {
            return get_value(launch::sync, reset, ec);
        }
        template <typename T>
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        T get_value_sync(error_code& ec = throws) const
        {
            return get_value(launch::sync, ec);
        }
#endif
    };

    /// Return all counters matching the given name (with optional wildcards).
    HPX_API_EXPORT std::vector<performance_counter> discover_counters(
        std::string const& name, error_code& ec = throws);
}}

#endif
