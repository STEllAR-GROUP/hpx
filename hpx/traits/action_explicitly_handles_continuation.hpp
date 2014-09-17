//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_ACTION_EXPLICITLY_HANDLES_CONTINUATION_APR_03_2014_0158PM)
#define HPX_TRAITS_ACTION_EXPLICITLY_HANDLES_CONTINUATION_APR_03_2014_0158PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <boost/type_traits/is_same.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for figuring out whether an action explicitly
    // handles its continuation
    namespace detail
    {
        // Actions which take continuation_type as their first argument are
        // assumed to handle their continuations explicitly.
        template <typename Arguments, int N = util::tuple_size<Arguments>::value>
        struct action_explicitly_handles_continuation
          : boost::is_same<
                typename util::tuple_element<0, Arguments>::type,
                actions::continuation_type>
        {};

        // nullary actions can't explicitly handle their continuations.
        template <typename Arguments>
        struct action_explicitly_handles_continuation<Arguments, 0>
          : boost::mpl::false_
        {};
    }

    template <typename Arguments, typename Enable>
    struct action_explicitly_handles_continuation
      : detail::action_explicitly_handles_continuation<Arguments>
    {};

    template <typename Action>
    struct action_explicitly_handles_continuation<Action, typename Action::type>
      : action_explicitly_handles_continuation<typename Action::type>
    {};
}}

#endif

