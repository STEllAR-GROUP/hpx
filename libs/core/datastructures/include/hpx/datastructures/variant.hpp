//  Copyright (c) 2021-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX17_COPY_ELISION)
#include <hpx/datastructures/variant_helper.hpp>
#include <variant>

namespace hpx {

    HPX_CXX_CORE_EXPORT using std::get;
    HPX_CXX_CORE_EXPORT using std::holds_alternative;
    HPX_CXX_CORE_EXPORT using std::monostate;
    HPX_CXX_CORE_EXPORT using std::variant;
    HPX_CXX_CORE_EXPORT using std::visit;
}    // namespace hpx

#else

#include <hpx/datastructures/detail/variant.hpp>

namespace hpx {

    HPX_CXX_CORE_EXPORT using hpx::variant_ns::get;
    HPX_CXX_CORE_EXPORT using hpx::variant_ns::holds_alternative;
    HPX_CXX_CORE_EXPORT using hpx::variant_ns::monostate;
    HPX_CXX_CORE_EXPORT using hpx::variant_ns::variant;
    HPX_CXX_CORE_EXPORT using hpx::variant_ns::visit;
}    // namespace hpx

#endif
