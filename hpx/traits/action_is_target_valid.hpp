//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_IS_TARGET_VALID_MAR_10_2014_1103AM)
#define HPX_TRAITS_ACTION_IS_TARGET_VALID_MAR_10_2014_1103AM

#include <hpx/runtime/naming/id_type.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for Action::is_target_valid
    template <typename Action, typename Enable>
    struct action_is_target_valid
    {
        static bool call(naming::id_type const& id)
        {
            return Action::is_target_valid(id);
        }
    };
}}

#endif

