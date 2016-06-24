//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_CAPABILITY_PROVIDER_JUN_09_2013_1257PM)
#define HPX_TRAITS_ACTION_CAPABILITY_PROVIDER_JUN_09_2013_1257PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SECURITY)
#include <hpx/components/security/capability.hpp>
#include <hpx/runtime/naming/address.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    template <typename Action, typename Enable = void>
    struct action_capability_provider
    {
        // return the required capabilities to invoke the given action
        static components::security::capability call(
            naming::address::address_type /*lva*/)
        {
            // by default actions don't require any capabilities to
            // be invoked
            return components::security::capability();
        }
    };
}}

#endif
#endif

