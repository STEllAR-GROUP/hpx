//  Copyright (c) 2019 The STE||AR GROUP
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion/evaluate_assert.hpp>
#include <hpx/assertion/force_linking.hpp>

namespace hpx { namespace assertion {
    // reference all symbols that have to be explicitly linked with the core
    // library
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper
        {
#if defined(HPX_DEBUG)
            &detail::handle_assert
#endif
        };
        return helper;
    }
}}    // namespace hpx::assertion
