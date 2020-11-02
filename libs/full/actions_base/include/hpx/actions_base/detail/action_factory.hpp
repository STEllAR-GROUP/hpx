//  Copyright (c)      2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/actions_base/actions_base_fwd.hpp>
#include <hpx/actions_base/actions_base_support.hpp>
#include <hpx/preprocessor/stringize.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hpx { namespace actions { namespace detail {

    struct action_registry
    {
    public:
        HPX_NON_COPYABLE(action_registry);

    public:
        using ctor_t = base_action* (*) (bool);
        using typename_to_ctor_t = std::unordered_map<std::string, ctor_t>;
        using typename_to_id_t = std::unordered_map<std::string, std::uint32_t>;
        using cache_t = std::vector<ctor_t>;

        HPX_STATIC_CONSTEXPR std::uint32_t invalid_id = ~0;

        HPX_EXPORT action_registry();
        HPX_EXPORT void register_factory(
            std::string const& type_name, ctor_t ctor);
        HPX_EXPORT void register_typename(
            std::string const& type_name, std::uint32_t id);
        HPX_EXPORT void fill_missing_typenames();
        HPX_EXPORT std::uint32_t try_get_id(std::string const& type_name) const;
        HPX_EXPORT std::vector<std::string> get_unassigned_typenames() const;

        HPX_EXPORT static std::uint32_t get_id(std::string const& type_name);
        HPX_EXPORT static base_action* create(
            std::uint32_t id, bool, std::string const* name = nullptr);

        HPX_EXPORT static action_registry& instance();

        void cache_id(std::uint32_t id, ctor_t ctor);
        std::string collect_registered_typenames();

        std::uint32_t max_id_;
        typename_to_ctor_t typename_to_ctor_;
        typename_to_id_t typename_to_id_;
        cache_t cache_;
    };

    template <std::uint32_t Id>
    HPX_ALWAYS_EXPORT std::string get_action_name_id();

    template <std::uint32_t Id>
    struct add_constant_entry
    {
    public:
        HPX_NON_COPYABLE(add_constant_entry);

    public:
        add_constant_entry();
        static add_constant_entry instance;
    };

    template <std::uint32_t Id>
    add_constant_entry<Id> add_constant_entry<Id>::instance;

    template <std::uint32_t Id>
    add_constant_entry<Id>::add_constant_entry()
    {
        action_registry::instance().register_typename(
            get_action_name_id<Id>(), Id);
    }
}}}    // namespace hpx::actions::detail

#define HPX_REGISTER_ACTION_FACTORY_ID(Name, Id)                               \
    namespace hpx { namespace actions { namespace detail {                     \
                template <>                                                    \
                HPX_ALWAYS_EXPORT std::string get_action_name_id<Id>()         \
                {                                                              \
                    return HPX_PP_STRINGIZE(Name);                             \
                }                                                              \
                template add_constant_entry<Id>                                \
                    add_constant_entry<Id>::instance;                          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/

#else

#define HPX_REGISTER_ACTION_FACTORY_ID(Name, Id) /**/

#endif
