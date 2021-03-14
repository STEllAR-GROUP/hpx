//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components/components_fwd.hpp>
#include <hpx/runtime_configuration/component_factory_base.hpp>

#include <cstddef>
#include <string>

namespace hpx {

    /// \namespace components
    namespace components {

        ///////////////////////////////////////////////////////////////////////
        template <typename Component>
        struct component_factory;

        class runtime_support;

        namespace stubs {

            struct runtime_support;
        }    // namespace stubs

        namespace server {

            class HPX_EXPORT runtime_support;
        }    // namespace server

    }    // namespace components

    HPX_EXPORT components::server::runtime_support* get_runtime_support_ptr();
}    // namespace hpx
