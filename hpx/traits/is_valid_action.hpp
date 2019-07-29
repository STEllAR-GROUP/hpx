//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_VALID_ACTION_JUN_28_2015_0133PM)
#define HPX_TRAITS_IS_VALID_ACTION_JUN_28_2015_0133PM

#include <hpx/config.hpp>
#include <hpx/type_support/decay.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    // verify that given Action is actually supported by the given Component
    template <typename Action, typename Component, typename Enable = void>
    struct is_valid_action
      : std::is_same<
            typename util::decay<typename Action::component_type>::type,
            typename util::decay<Component>::type>
    {};
}}

#endif

