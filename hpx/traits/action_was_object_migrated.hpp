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

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    template <typename Action, typename Enable>
    struct action_was_object_migrated
    {
        // returns whether target was migrated to another locality
        static std::pair<bool, components::pinned_ptr>
        call(hpx::id_type const& id, naming::address::address_type lva)
        {
            // by default we forward this to the component type
            typedef typename Action::component_type component_type;
            return component_type::was_object_migrated(id, lva);
        }
    };
}}

#endif

