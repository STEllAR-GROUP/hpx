//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_ACTION_APR_15_2012_0601PM)
#define HPX_TRAITS_IS_ACTION_APR_15_2012_0601PM

#include <hpx/traits.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/has_xxx.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

namespace hpx { namespace traits
{
    namespace detail
    {
        BOOST_MPL_HAS_XXX_TRAIT_DEF(action_tag)
    }

    template <typename Action, typename Enable>
    struct is_action
      : detail::has_action_tag<Action>
    {};

    template <typename Action>
    struct is_action<Action, typename Action::type>
      : is_action<typename Action::type>
    {};
}}

#endif

