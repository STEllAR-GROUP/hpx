////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B08244B4_3831_436F_9F72_3E82FFAF03E8)
#define HPX_B08244B4_3831_436F_9F72_3E82FFAF03E8

#include <hpx/config.hpp>
#include <hpx/traits/component_type_database.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <utility>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class simple_component;

    template <typename Component>
    class abstract_simple_component_base
      : private traits::detail::simple_component_tag
    {
    private:
        typedef simple_component<Component> outer_wrapping_type;

    public:
        virtual ~abstract_simple_component_base() {}

        typedef Component base_type_holder;

        static component_type get_component_type()
        {
            return hpx::components::get_component_type<outer_wrapping_type>();
        }

        static void set_component_type(component_type t)
        {
            hpx::components::set_component_type<outer_wrapping_type>(t);
        }

        // This component type requires valid id for its actions to be invoked
        static bool is_target_valid(naming::id_type const& id)
        {
            return !naming::is_locality(id);
        }

        /// This is the default hook implementation for decorate_action which
        /// does no hooking at all.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address::address_type, F && f)
        {
            return std::forward<F>(f);
        }

        /// This is the default hook implementation for schedule_thread which
        /// forwards to the default scheduler.
        static void schedule_thread(naming::address::address_type,
            threads::thread_init_data& data,
            threads::thread_state_enum initial_state)
        {
            hpx::threads::register_work_plain(data, initial_state); //-V106
        }
    };

    template <typename Component>
    class abstract_component_base
      : public abstract_simple_component_base<Component>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Derived>
    class managed_component;

    template <typename Component, typename Wrapper>
    class abstract_managed_component_base
      : private traits::detail::managed_component_tag
    {
    public:
        typedef managed_component<Component, Wrapper> wrapping_type;

        virtual ~abstract_managed_component_base() {}

        typedef Component base_type_holder;

        static component_type get_component_type()
        {
            return hpx::components::get_component_type<wrapping_type>();
        }

        static void set_component_type(component_type t)
        {
            hpx::components::set_component_type<wrapping_type>(t);
        }

        // This component type requires valid id for its actions to be invoked
        static bool is_target_valid(naming::id_type const& id)
        {
            return !naming::is_locality(id);
        }

        /// This is the default hook implementation for decorate_action which
        /// does no hooking at all.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address::address_type, F && f)
        {
            return std::forward<F>(f);
        }

        /// This is the default hook implementation for schedule_thread which
        /// forwards to the default scheduler.
        static void schedule_thread(naming::address::address_type,
            threads::thread_init_data& data,
            threads::thread_state_enum initial_state)
        {
            hpx::threads::register_work_plain(data, initial_state); //-V106
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class fixed_component;

    template <typename Component>
    class abstract_fixed_component_base
      : private traits::detail::fixed_component_tag
    {
    private:
        typedef fixed_component<Component> outer_wrapping_type;

    public:
        virtual ~abstract_fixed_component_base() {}

        typedef Component base_type_holder;

        static component_type get_component_type()
        {
            return hpx::components::get_component_type<outer_wrapping_type>();
        }

        static void set_component_type(component_type t)
        {
            hpx::components::set_component_type<outer_wrapping_type>(t);
        }

        // This component type requires valid id for its actions to be invoked
        static bool is_target_valid(naming::id_type const& id)
        {
            return !naming::is_locality(id);
        }

        /// This is the default hook implementation for decorate_action which
        /// does no hooking at all.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address::address_type, F && f)
        {
            return std::forward<F>(f);
        }

        /// This is the default hook implementation for schedule_thread which
        /// forwards to the default scheduler.
        static void schedule_thread(naming::address::address_type,
            threads::thread_init_data& data,
            threads::thread_state_enum initial_state)
        {
            hpx::threads::register_work_plain(data, initial_state); //-V106
        }
    };
}}

#endif // HPX_B08244B4_3831_436F_9F72_3E82FFAF03E8

