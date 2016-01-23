//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_WAS_OBJECT_MIGRATED_JAN_22_2016_1115AM)
#define HPX_TRAITS_ACTION_WAS_OBJECT_MIGRATED_JAN_22_2016_1115AM

#include <hpx/config.hpp>
#include <hpx/traits.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>

#include <utility>
#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    namespace detail
    {
        struct was_object_migrated_helper
        {
            // by default we return an empty value
            template <typename Action>
            static std::pair<bool, components::pinned_ptr>
            call(wrap_int, hpx::id_type const&, naming::address::address_type)
            {
                return std::make_pair(false, components::pinned_ptr());
            }

            // forward the call if the component implements the function
            template <typename Action>
            static auto
            call(int, hpx::id_type const& id, naming::address::address_type lva)
            ->  decltype(
                    typename Action::component_type::was_object_migrated(id, lva)
                )
            {
                typedef typename Action::component_type component_type;
                return component_type::was_object_migrated(id, lva);
            }
        };

        template <typename Action>
        std::pair<bool, components::pinned_ptr>
        call_was_object_migrated(hpx::id_type const& id,
            naming::address::address_type lva)
        {
            return was_object_migrated_helper::template call<Action>(0, id, lva);
        }
    }

    template <typename Action, typename Enable>
    struct action_was_object_migrated
    {
        // returns whether target was migrated to another locality
        static std::pair<bool, components::pinned_ptr>
        call(hpx::id_type const& id, naming::address::address_type lva)
        {
            return detail::call_was_object_migrated<Action>(id, lva);
        }
    };
}}

#endif

