//  Copyright (c) 2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/thread_executors/embedded_thread_pool_executors.hpp>

namespace hpx { namespace thread_executors {

    struct force_linking_helper
    {
        void (*dummy_ptr)();
    };

    force_linking_helper& force_linking();
}}    // namespace hpx::thread_executors
