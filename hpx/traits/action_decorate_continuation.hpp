//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_DECORATE_CONTINUATION_MAR_30_2014_0725PM)
#define HPX_TRAITS_ACTION_DECORATE_CONTINUATION_MAR_30_2014_0725PM

#include <hpx/runtime/actions/continuation_fwd.hpp>
#include <hpx/traits/action_continuation.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action capabilities
    template <typename Action, typename Enable = void>
    struct action_decorate_continuation
    {
        using continuation_type =
            typename traits::action_continuation<Action>::type;

        static bool call(continuation_type& /*cont*/)
        {
            // by default we do nothing
            return false; // continuation has not been modified
        }
    };
}}

#endif
