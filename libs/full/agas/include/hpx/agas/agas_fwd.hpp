//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/agas_base/agas_fwd.hpp>

namespace hpx {
    namespace agas {

        struct HPX_EXPORT addressing_service;
    }    // namespace agas

    namespace naming {

        // FIXME: obsolete name, replace with agas::addressing_serve
        using resolver_client = agas::addressing_service;

        HPX_EXPORT agas::addressing_service& get_agas_client();
        HPX_EXPORT agas::addressing_service* get_agas_client_ptr();
    }    // namespace naming
}    // namespace hpx
