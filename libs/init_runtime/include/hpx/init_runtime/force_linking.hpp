//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/hpx_finalize.hpp>

namespace hpx { namespace init_runtime {

    struct force_linking_helper
    {
        int (*finalize)(double, double, error_code&);
#if defined(HPX_WINDOWS)
        void (*init_winsocket)();
#endif
    };

    force_linking_helper& force_linking();
}}    // namespace hpx::init_runtime
