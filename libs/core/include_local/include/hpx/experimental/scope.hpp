//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX26_EXPERIMENTAL_SCOPE)

#include <experimental/scope>

namespace hpx::experimental {

    using std::experimental::scope_exit;
    using std::experimental::scope_failure;
    using std::experimental::scope_success;
}    // namespace hpx::experimental

#else

#include <hpx/functional/experimental/scope_exit.hpp>
#include <hpx/functional/experimental/scope_fail.hpp>
#include <hpx/functional/experimental/scope_success.hpp>

#endif
