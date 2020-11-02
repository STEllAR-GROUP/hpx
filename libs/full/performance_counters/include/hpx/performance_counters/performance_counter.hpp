//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <hpx/performance_counters/counters_fwd.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters {

    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT performance_counter
      : components::client_base<performance_counter,
            server::base_performance_counter>
    {
        using base_type = components::client_base<performance_counter,
            server::base_performance_counter>;

        performance_counter() = default;

        performance_counter(std::string const& name);

        performance_counter(
            std::string const& name, hpx::id_type const& locality);

        performance_counter(id_type const& id)
          : base_type(id)
        {
        }

        performance_counter(future<id_type>&& id)
          : base_type(std::move(id))
        {
        }

        performance_counter(hpx::future<performance_counter>&& c)
          : base_type(std::move(c))
        {
        }

        ///////////////////////////////////////////////////////////////////////
        future<counter_info> get_info() const;
        counter_info get_info(
            launch::sync_policy, error_code& ec = throws) const;

        future<counter_value> get_counter_value(bool reset = false);
        counter_value get_counter_value(
            launch::sync_policy, bool reset = false, error_code& ec = throws);

        future<counter_value> get_counter_value() const;
        counter_value get_counter_value(
            launch::sync_policy, error_code& ec = throws) const;

        future<counter_values_array> get_counter_values_array(
            bool reset = false);
        counter_values_array get_counter_values_array(
            launch::sync_policy, bool reset = false, error_code& ec = throws);

        future<counter_values_array> get_counter_values_array() const;
        counter_values_array get_counter_values_array(
            launch::sync_policy, error_code& ec = throws) const;

        ///////////////////////////////////////////////////////////////////////
        future<bool> start();
        bool start(launch::sync_policy, error_code& ec = throws);

        future<bool> stop();
        bool stop(launch::sync_policy, error_code& ec = throws);

        future<void> reset();
        void reset(launch::sync_policy, error_code& ec = throws);

        future<void> reinit(bool reset = true);
        void reinit(
            launch::sync_policy, bool reset = true, error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        future<std::string> get_name() const;
        std::string get_name(
            launch::sync_policy, error_code& ec = throws) const;

    private:
        template <typename T>
        static T extract_value(future<counter_value>&& value)
        {
            return value.get().get_value<T>();
        }

    public:
        template <typename T>
        future<T> get_value(bool reset = false)
        {
            return get_counter_value(reset).then(hpx::launch::sync,
                util::bind_front(&performance_counter::extract_value<T>));
        }
        template <typename T>
        T get_value(
            launch::sync_policy, bool reset = false, error_code& ec = throws)
        {
            return get_counter_value(launch::sync, reset).get_value<T>(ec);
        }

        template <typename T>
        future<T> get_value() const
        {
            return get_counter_value().then(hpx::launch::sync,
                util::bind_front(&performance_counter::extract_value<T>));
        }
        template <typename T>
        T get_value(launch::sync_policy, error_code& ec = throws) const
        {
            return get_counter_value(launch::sync).get_value<T>(ec);
        }
    };

    // Return all counters matching the given name (with optional wild cards).
    HPX_EXPORT std::vector<performance_counter> discover_counters(
        std::string const& name, error_code& ec = throws);
}}    // namespace hpx::performance_counters
