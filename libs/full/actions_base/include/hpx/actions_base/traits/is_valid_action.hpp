//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <type_traits>

namespace hpx { namespace traits {

    // verify that given Action is actually supported by the given Component
    template <typename Action, typename Component, typename Enable = void>
    struct is_valid_action
      : std::is_same<typename std::decay<typename Action::component_type>::type,
            typename std::decay<Component>::type>
    {
    };
}}    // namespace hpx::traits


