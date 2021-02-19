//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components/components_fwd.hpp>

#include <cstddef>
#include <string>

namespace hpx {

    enum logging_destination
    {
        destination_hpx = 0,
        destination_timing = 1,
        destination_agas = 2,
        destination_parcel = 3,
        destination_app = 4,
        destination_debuglog = 5
    };

    /// \namespace components
    namespace components {

        ///////////////////////////////////////////////////////////////////////
        struct HPX_EXPORT component_factory_base;

        template <typename Component>
        struct component_factory;

        class runtime_support;
        class memory;


        namespace stubs {
            struct runtime_support;
            struct memory;
        }    // namespace stubs

        namespace server {
            class HPX_EXPORT runtime_support;
            class HPX_EXPORT memory;
        }    // namespace server

        HPX_EXPORT void console_logging(logging_destination dest,
            std::size_t level, std::string const& msg);
        HPX_EXPORT void cleanup_logging();
        HPX_EXPORT void activate_logging();
    }    // namespace components

    HPX_EXPORT components::server::runtime_support* get_runtime_support_ptr();
}    // namespace hpx
