//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/basic_execution/default_context.hpp>
#include <hpx/functional/unique_function.hpp>

#include <thread>

namespace hpx { namespace basic_execution {
    void default_context::post(hpx::util::unique_function_nonser<void()> f) const
    {
        std::thread t(std::move(f));
        t.detach();
    }
}}
