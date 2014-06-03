//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_MAY_REQUIRE_ID_SPLITTING_JAN_26_2014_1128AM)
#define HPX_TRAITS_MAY_REQUIRE_ID_SPLITTING_JAN_26_2014_1128AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/always_void.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action stack size
    template <typename Action, typename Enable>
    struct action_may_require_id_splitting
    {
        // return a new instance of a serialization filter
        template <typename Arguments>
        static bool call(Arguments const& /*act*/)
        {
            // by default actions are assumed to require id-splitting
            return true;
        }
    };

    template <typename Action>
    struct action_may_require_id_splitting<Action
      , typename util::always_void<typename Action::type>::type>
      : action_may_require_id_splitting<typename Action::type>
    {};
}}

#endif

