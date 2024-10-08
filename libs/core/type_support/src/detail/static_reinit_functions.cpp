//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/type_support/detail/static_reinit_functions.hpp>

#include <functional>

namespace hpx::util::detail {

    void (*reinit_register)(std::function<void()> const& construct,
        std::function<void()> const& destruct) = nullptr;

    // Invoke all globally registered construction functions
    void (*reinit_construct)() = nullptr;

    // Invoke all globally registered destruction functions
    void (*reinit_destruct)() = nullptr;
}    // namespace hpx::util::detail
