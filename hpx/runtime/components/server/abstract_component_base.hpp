////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2013-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B08244B4_3831_436F_9F72_3E82FFAF03E8)
#define HPX_B08244B4_3831_436F_9F72_3E82FFAF03E8

#include <hpx/config.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/threads/thread_init_data.hpp>
#include <hpx/traits/is_component.hpp>

#include <utility>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class simple_component;

    template <typename Component>
    class abstract_simple_component_base
      : private traits::detail::component_tag
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
    };
}}

#endif // HPX_B08244B4_3831_436F_9F72_3E82FFAF03E8

