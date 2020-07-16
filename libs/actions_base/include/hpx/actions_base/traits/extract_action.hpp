//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

namespace hpx { namespace traits {

    // This template meta function can be used to extract the action type.
    template <typename Action, typename Enable = void>
    struct extract_action
    {
        using type = typename Action::derived_type;
        using result_type = typename type::result_type;
        using local_result_type = typename type::local_result_type;
        using remote_result_type = typename type::remote_result_type;
    };
}}    // namespace hpx::traits

#if defined(HPX_HAVE_ACTION_BASE_COMPATIBILITY)
#include <hpx/type_support/always_void.hpp>

namespace hpx { namespace traits {

    template <typename Action>
    struct extract_action<Action,
        typename util::always_void<typename Action::type>::type>
      : extract_action<typename Action::type>
    {
    };
}}    // namespace hpx::traits

#endif
