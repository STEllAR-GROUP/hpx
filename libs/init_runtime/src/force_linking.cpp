//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/hpx_finalize.hpp>
#include <hpx/init_runtime/force_linking.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/hpx_main_winsocket.hpp>
#endif

namespace hpx { namespace init_runtime {
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper
        {
            &hpx::finalize
#if defined(HPX_WINDOWS)
                ,
                &hpx::detail::init_winsocket
#endif
        };
        return helper;
    }
}}    // namespace hpx::init_runtime
