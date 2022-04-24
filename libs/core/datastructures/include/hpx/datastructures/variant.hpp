//  Copyright (c) 2021 Hartmut Kaiser
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

    using std::get;
    using std::holds_alternative;
    using std::monostate;
    using std::variant;
    using std::visit;
}    // namespace hpx

#else

#include <hpx/datastructures/detail/variant.hpp>

namespace hpx {

    using hpx::variant_ns::get;
    using hpx::variant_ns::holds_alternative;
    using hpx::variant_ns::monostate;
    using hpx::variant_ns::variant;
    using hpx::variant_ns::visit;
}    // namespace hpx

#endif
