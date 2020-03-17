//  Copyright (c) 2019 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local_lcos/composable_guard.hpp>
#include <hpx/local_lcos/force_linking.hpp>

namespace hpx { namespace local_lcos {

    force_linking_helper& force_linking()
    {
        static force_linking_helper helper{&lcos::local::detail::free};
        return helper;
    }
}}    // namespace hpx::local_lcos
