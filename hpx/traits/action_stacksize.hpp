//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_STACKSIZE_SEP_03_2012_1136AM)
#define HPX_TRAITS_ACTION_STACKSIZE_SEP_03_2012_1136AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/always_void.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for action stack size
    template <typename Action, typename Enable>
    struct action_stacksize
    {
        enum { value = threads::thread_stacksize_default };
    };

    template <typename Action>
    struct action_stacksize<Action
      , typename util::always_void<typename Action::type>::type>
      : action_stacksize<typename Action::type>
    {};
}}

#endif

