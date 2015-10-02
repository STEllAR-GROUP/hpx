//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_DOES_TERMINATION_DETECTION_MAR_21_2014_0818PM)
#define HPX_TRAITS_ACTION_DOES_TERMINATION_DETECTION_MAR_21_2014_0818PM

#include <hpx/util/always_void.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for Action::does_termination_detection
    template <typename Action, typename Enable>
    struct action_does_termination_detection
    {
        static bool call()
        {
            return false;
        }
    };

    template <typename Action>
    struct action_does_termination_detection<Action
      , typename util::always_void<typename Action::type>::type>
      : action_does_termination_detection<typename Action::type>
    {};
}}

#endif

