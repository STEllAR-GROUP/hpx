//  Copyright (c) 2019 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/executors/current_executor.hpp>
#include <hpx/executors/force_linking.hpp>

namespace hpx { namespace executors {

    // reference all symbols that have to be explicitly linked with the core
    // library
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper{&hpx::this_thread::get_executor};
        return helper;
    }
}}    // namespace hpx::executors
