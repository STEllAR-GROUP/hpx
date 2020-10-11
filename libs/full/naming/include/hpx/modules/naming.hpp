//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/naming_base.hpp>

namespace hpx {

    namespace naming {

        struct HPX_EXPORT gid_type;
        struct HPX_EXPORT id_type;
        struct HPX_EXPORT address;
    }    // namespace naming

    // Pulling important types into the main namespace
    using naming::id_type;
}    // namespace hpx
