//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <exception>

namespace hpx { namespace execution { namespace experimental {
    struct sink_receiver
    {
        void set_error(std::exception_ptr) noexcept {}
        void set_done() noexcept {};
        void set_value() noexcept {}
    };
}}}    // namespace hpx::execution::experimental
