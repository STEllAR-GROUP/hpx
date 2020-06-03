////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/modules/async_distributed.hpp>
#include <hpx/include/client.hpp>

#include <cstdint>

#include "server/simple_mobile_object.hpp"

namespace hpx { namespace test
{

struct simple_mobile_object
  : components::client_base<
        simple_mobile_object, server::simple_mobile_object
    >
{
    typedef components::client_base<
        simple_mobile_object, server::simple_mobile_object
    > base_type;

  public:
    typedef server::simple_mobile_object server_type;

    /// Create a new component on the target locality.
    explicit simple_mobile_object(
        naming::id_type const& locality
        )
      : base_type(stub_type::create_async(locality))
    {
    }

    std::uint64_t get_lva()
    {
        typedef server_type::get_lva_action action_type;
        return hpx::async<action_type>(this->get_id()).get();
    }
};

}}


