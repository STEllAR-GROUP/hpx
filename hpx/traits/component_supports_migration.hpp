//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_SUPPORTS_MIGRATION_FEB_10_2016_1252PM)
#define HPX_TRAITS_ACTION_SUPPORTS_MIGRATION_FEB_10_2016_1252PM

#include <hpx/config.hpp>
#include <hpx/traits.hpp>

#include <utility>
#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for component capabilities
    namespace detail
    {
        struct supports_migration_helper
        {
            // by default we return 'false' (component does not support
            // migration)
            template <typename Component>
            static HPX_CONSTEXPR bool call(wrap_int)
            {
                return false;
            }

            // forward the call if the component implements the function
            template <typename Component>
            static HPX_CONSTEXPR auto call(int)
            ->  decltype(Component::supports_migration())
            {
                return Component::supports_migration();
            }
        };

        template <typename Component>
        HPX_CONSTEXPR bool call_supports_migration()
        {
            return supports_migration_helper::template call<Component>(0);
        }
    }

    template <typename Component, typename Enable>
    struct component_supports_migration
    {
        // returns whether target supports migration
        static HPX_CONSTEXPR bool call()
        {
            return detail::call_supports_migration<Component>();
        }
    };
}}

#endif

