//  Copyright (c)      2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_ACTIONS_DETAIL_ACTION_FACTORY_HPP
#define HPX_ACTIONS_DETAIL_ACTION_FACTORY_HPP

#include <hpx/config.hpp>

#include <hpx/runtime/actions_fwd.hpp>
#include <hpx/runtime/actions/action_support.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hpx { namespace actions { namespace detail
{
    struct action_registry
    {
        HPX_NON_COPYABLE(action_registry);

    public:
        typedef base_action* (*ctor_t)(bool);
        typedef std::unordered_map<std::string, ctor_t> typename_to_ctor_t;
        typedef std::unordered_map<std::string, std::uint32_t> typename_to_id_t;
        typedef std::vector<ctor_t> cache_t;

        HPX_STATIC_CONSTEXPR std::uint32_t invalid_id = ~0;

        HPX_EXPORT action_registry();
        HPX_EXPORT void register_factory(std::string const& type_name, ctor_t ctor);
        HPX_EXPORT void register_typename(std::string const& type_name, std::uint32_t id);
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

    template <typename Action>
    struct register_action
    {
        HPX_NON_COPYABLE(register_action);

    public:
        register_action();
        static base_action* create(bool);
        register_action& instantiate();

        static register_action instance;
    };

    template <typename Action>
    register_action<Action> register_action<Action>::instance;

    template <typename Action>
    register_action<Action>::register_action()
    {
        action_registry::instance().register_factory(
            hpx::actions::detail::get_action_name<Action>(),
            &create);
    }

    template <typename Action>
    base_action* register_action<Action>::create(bool has_continuation)
    {
        if (has_continuation)
            return new transfer_continuation_action<Action>();
        else
            return new transfer_action<Action>();
    }

    template <typename Action>
    register_action<Action>& register_action<Action>::instantiate()
    {
        return *this;
    }

    template <std::uint32_t Id>
    std::string get_action_name_id();

    template <std::uint32_t Id>
    struct add_constant_entry
    {
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
}}}

#define HPX_REGISTER_ACTION_FACTORY_ID(Name, Id)                                \
    namespace hpx { namespace actions { namespace detail {                      \
        template <> std::string get_action_name_id<Id>()                        \
        {                                                                       \
            return BOOST_PP_STRINGIZE(Name);                                    \
        }                                                                       \
        template add_constant_entry<Id> add_constant_entry<Id>::instance;       \
    }}}                                                                         \
/**/

#endif
