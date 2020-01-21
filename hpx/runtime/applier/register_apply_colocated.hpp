//  Copyright (c) 2014 thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_APPLIER_REGISTER_APPLY_COLOCATED_MAR_09_2014_1214PM
#define HPX_RUNTIME_APPLIER_REGISTER_APPLY_COLOCATED_MAR_09_2014_1214PM

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/util/functional/colocated_helpers.hpp>

namespace hpx { namespace detail
{
    template <typename Tuple>
    struct apply_colocated_bound_tuple;

    template <typename ...Ts>
    struct apply_colocated_bound_tuple<util::tuple<Ts...> >
    {
        typedef
            util::tuple<
                hpx::util::detail::bound<
                    hpx::util::functional::extract_locality
                  , hpx::util::detail::placeholder<2ul>
                  , hpx::id_type
                >
              , Ts...
            >
            type;
    };
}}

#define HPX_REGISTER_APPLY_COLOCATED_DECLARATION(Action, Name)                \

/*
    HPX_UTIL_REGISTER_UNIQUE_FUNCTION_DECLARATION(                            \
        void (hpx::naming::id_type, hpx::naming::id_type)                     \
      , (hpx::util::functional::detail::apply_continuation_impl<              \
            hpx::util::detail::bound_action<                                  \
                Action                                                        \
              , hpx::detail::apply_colocated_bound_tuple<                     \
                    Action ::arguments_type                                   \
                >::type                                                       \
            >                                                                 \
        >)                                                                    \
      , Name                                                                  \
    );                                                                        \
    */
/**/

#define HPX_REGISTER_APPLY_COLOCATED(action, name)                            \

/*
    HPX_UTIL_REGISTER_UNIQUE_FUNCTION(                                        \
        void (hpx::naming::id_type, hpx::naming::id_type)                     \
      , (hpx::util::functional::detail::apply_continuation_impl<              \
            hpx::util::detail::bound_action<                                  \
                action                                                        \
              , hpx::detail::apply_colocated_bound_tuple<                     \
                    action::arguments_type                                    \
                >::type                                                       \
            >                                                                 \
        >)                                                                    \
      , name                                                                  \
    );                                                                        \
    */
/**/

#endif
