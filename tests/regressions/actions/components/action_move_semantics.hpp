//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/include/components.hpp>

#include "server/action_move_semantics.hpp"

#include <utility>

namespace hpx { namespace test
{
    struct action_move_semantics
      : components::client_base<action_move_semantics,
            server::action_move_semantics>
    {
        typedef components::client_base<
            action_move_semantics, server::action_move_semantics
        > base_type;

        action_move_semantics() = default;
        explicit action_move_semantics(naming::id_type const& id)
          : base_type(id)
        {}
        action_move_semantics(hpx::future<naming::id_type> && id)
          : base_type(std::move(id))
        {}
    };
}}

#endif
