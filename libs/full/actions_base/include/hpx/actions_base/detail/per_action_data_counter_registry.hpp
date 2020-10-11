//  Copyright (c) 2016-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS) && defined(HPX_HAVE_NETWORKING)
#include <hpx/functional/function.hpp>
#include <hpx/hashing/jenkins_hash.hpp>
#include <hpx/type_support/static.hpp>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace actions { namespace detail {

    class HPX_EXPORT per_action_data_counter_registry
    {
    public:
        HPX_NON_COPYABLE(per_action_data_counter_registry);

    public:
        using counter_function_type = util::function_nonser<std::int64_t(bool)>;
        using map_type =
            std::unordered_set<std::string, hpx::util::jenkins_hash>;

        enum per_action_counter_type
        {
            num_parcels = 0,
            num_messages,
            total_time,
            total_serialization_time,
            total_bytes,
            total_raw_bytes,
            total_buffer_allocate_time
        };

        per_action_data_counter_registry() = default;

        static per_action_data_counter_registry& instance();

        void register_class(std::string action);

        counter_function_type get_counter(std::string const& action,
            hpx::util::function_nonser<std::int64_t(
                std::string const&, bool)> const& f) const;

        map_type const& registered_counters() const
        {
            return map_;
        }

    private:
        struct tag
        {
        };
        friend struct hpx::util::static_<per_action_data_counter_registry, tag>;

        map_type map_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    void register_per_action_data_counter_types(
        per_action_data_counter_registry& registry);

    template <typename Action>
    struct register_per_action_data_counters
    {
        register_per_action_data_counters()
        {
            register_per_action_data_counter_types<Action>(
                per_action_data_counter_registry::instance());
        }

        static register_per_action_data_counters instance;
    };

    template <typename Action>
    register_per_action_data_counters<Action>
        register_per_action_data_counters<Action>::instance;
}}}    // namespace hpx::actions::detail

#define HPX_REGISTER_PER_ACTION_DATA_COUNTER_TYPES(Action)                     \
    namespace hpx { namespace actions { namespace detail {                     \
                template register_per_action_data_counters<Action>             \
                    register_per_action_data_counters<Action>::instance;       \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/

#include <hpx/config/warnings_suffix.hpp>

#else

#define HPX_REGISTER_PER_ACTION_DATA_COUNTER_TYPES(Action)

#endif
