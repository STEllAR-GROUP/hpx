//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_EXTRACT_ACTION_JUN_28_2015_0518PM)
#define HPX_TRAITS_EXTRACT_ACTION_JUN_28_2015_0518PM

#include <hpx/traits.hpp>
#include <hpx/util/always_void.hpp>

namespace hpx { namespace traits
{
    // This template meta function can be used to extract the action type, no
    // matter whether it got specified directly or by passing the
    // corresponding make_action<> specialization.
    template <typename Action, typename Enable>
    struct extract_action
    {
        typedef typename Action::derived_type type;
        typedef typename type::result_type result_type;
        typedef typename type::remote_result_type remote_result_type;
    };

    template <typename Action>
    struct extract_action<Action
      , typename util::always_void<typename Action::type>::type>
      : extract_action<typename Action::type>
    {};
}}

#endif

