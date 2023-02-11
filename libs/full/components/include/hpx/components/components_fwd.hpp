//  Copyright (c) 2017-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/components_base/components_base_fwd.hpp>

namespace hpx {

    /// \namespace components
    namespace components {

        template <typename Derived, typename Stub, typename ClientData = void>
        class client_base;
    }    // namespace components
}    // namespace hpx
