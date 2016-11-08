//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_ACTION_CONTINUATION_HPP
#define HPX_TRAITS_ACTION_CONTINUATION_HPP

#include <hpx/traits/extract_action.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Trait to determine the continuation type for an action
    template <typename Action, typename Enable = void>
    struct action_continuation
    {
        typedef
            typename hpx::traits::extract_action<Action>::type::continuation_type
            type;
    };
}}

#endif
