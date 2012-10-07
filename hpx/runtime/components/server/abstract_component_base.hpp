////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B08244B4_3831_436F_9F72_3E82FFAF03E8)
#define HPX_B08244B4_3831_436F_9F72_3E82FFAF03E8

#include <hpx/hpx_fwd.hpp>

namespace hpx { namespace components
{
    template <typename Component>
    class simple_component;

    template <typename Component>
    class abstract_simple_component_base
    {
      public:
        virtual ~abstract_simple_component_base() {}

        typedef simple_component<Component> wrapping_type;
        typedef Component base_type_holder;

        static component_type get_component_type()
        {
            return hpx::components::get_component_type<wrapping_type>();
        }

        static void set_component_type(component_type t)
        {
            hpx::components::set_component_type<wrapping_type>();
        }
    };

    template <typename Component, typename Derived>
    class managed_component;

    template <typename Component, typename Wrapper>
    class abstract_managed_component_base
    {
      public:
        virtual ~abstract_managed_component_base() {}

        typedef managed_component<Component, Wrapper> wrapping_type;
        typedef Component base_type_holder;

        static component_type get_component_type()
        {
            return hpx::components::get_component_type<wrapping_type>();
        }

        static void set_component_type(component_type t)
        {
            hpx::components::set_component_type<wrapping_type>();
        }
    };
}}

#endif // HPX_B08244B4_3831_436F_9F72_3E82FFAF03E8

