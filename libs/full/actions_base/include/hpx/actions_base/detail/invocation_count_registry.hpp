//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/hashing/jenkins_hash.hpp>
#include <hpx/type_support/static.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace actions { namespace detail {

    class HPX_EXPORT invocation_count_registry
    {
    public:
        HPX_NON_COPYABLE(invocation_count_registry);

    public:
        using get_invocation_count_type = std::int64_t (*)(bool);
        using map_type = std::unordered_map<std::string,
            get_invocation_count_type, hpx::util::jenkins_hash>;

        invocation_count_registry() = default;

        static invocation_count_registry& local_instance();
#if defined(HPX_HAVE_NETWORKING)
        static invocation_count_registry& remote_instance();
#endif

        void register_class(
            std::string const& name, get_invocation_count_type fun);

        get_invocation_count_type get_invocation_counter(
            std::string const& name) const;

        map_type const& registered_counters() const
        {
            return map_;
        }

    private:
        struct local_tag
        {
        };
        friend struct hpx::util::static_<invocation_count_registry, local_tag>;

#if defined(HPX_HAVE_NETWORKING)
        struct remote_tag
        {
        };
        friend struct hpx::util::static_<invocation_count_registry, remote_tag>;
#endif
        map_type map_;
    };

    template <typename Action>
    void register_local_action_invocation_count(
        invocation_count_registry& registry);

#if defined(HPX_HAVE_NETWORKING)
    template <typename Action>
    void register_remote_action_invocation_count(
        invocation_count_registry& registry);
#endif

    template <typename Action>
    struct register_action_invocation_count
    {
        register_action_invocation_count()
        {
            register_local_action_invocation_count<Action>(
                invocation_count_registry::local_instance());

#if defined(HPX_HAVE_NETWORKING)
            register_remote_action_invocation_count<Action>(
                invocation_count_registry::remote_instance());
#endif
        }

        static register_action_invocation_count instance;
    };

    template <typename Action>
    register_action_invocation_count<Action>
        register_action_invocation_count<Action>::instance;
}}}    // namespace hpx::actions::detail

#define HPX_REGISTER_ACTION_INVOCATION_COUNT(Action)                           \
    namespace hpx { namespace actions { namespace detail {                     \
                template register_action_invocation_count<Action>              \
                    register_action_invocation_count<Action>::instance;        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/

#include <hpx/config/warnings_suffix.hpp>
