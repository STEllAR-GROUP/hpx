//  Copyright (c) 2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/executors/current_executor.hpp>

namespace hpx { namespace executors {
    struct force_linking_helper
    {
        parallel::execution::current_executor (*get_executor)(error_code&);
    };

    force_linking_helper& force_linking();
}}    // namespace hpx::executors
