//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

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

        action_move_semantics() {}
        action_move_semantics(hpx::future<naming::id_type> && id)
          : base_type(std::move(id))
        {}
    };
}}


