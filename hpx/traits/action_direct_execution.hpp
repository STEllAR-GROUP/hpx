//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_ACTION_DIRECT_EXECUTION_HPP
#define HPX_TRAITS_ACTION_DIRECT_EXECUTION_HPP

#include <hpx/util/always_void.hpp>

#include <type_traits>

namespace hpx { namespace traits {
    template <typename Action, typename Enable = void>
    struct action_direct_execution
      : Action::direct_execution
    {};
}}

#define HPX_TRAITS_ACTION_DIRECT_EXECUTION(Action, DirectExecution)            \
    namespace hpx { namespace traits {                                         \
        template <>                                                            \
        struct action_direct_execution<Action>                                 \
          : std:: BOOST_PP_CAT(DirectExecution, _type)                         \
        {};                                                                    \
    }}                                                                         \
/**/

#endif
