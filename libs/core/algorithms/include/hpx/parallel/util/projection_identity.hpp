//  Copyright (c) 2015-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/identity.hpp>

namespace hpx::parallel::util {

    ///////////////////////////////////////////////////////////////////////////
    using projection_identity HPX_DEPRECATED_V(1, 9,
        "hpx::identity is deprecated, use "
        "hpx::identity instead") = hpx::identity;
}    // namespace hpx::parallel::util
