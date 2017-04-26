//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_ACTION_APR_15_2012_0601PM)
#define HPX_TRAITS_IS_ACTION_APR_15_2012_0601PM

#include <hpx/config.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename Action, typename Enable = void>
        struct is_action_impl
          : std::false_type
        {};

        template <typename Action>
        struct is_action_impl<Action,
            typename util::always_void<typename Action::action_tag>::type
        > : std::true_type
        {};
    }

    template <typename Action, typename Enable = void>
    struct is_action
      : detail::is_action_impl<typename util::decay<Action>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct is_bound_action
      : std::false_type
    {};
}}

#endif

