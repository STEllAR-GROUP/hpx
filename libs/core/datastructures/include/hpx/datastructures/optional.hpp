//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX17_COPY_ELISION) &&                                    \
    defined(HPX_HAVE_CXX17_OPTIONAL_COPY_ELISION)

#include <functional>    // for std::hash<std::optional<T>>
#include <optional>

namespace hpx {

    using std::bad_optional_access;
    using std::make_optional;
    using std::nullopt;
    using std::nullopt_t;
    using std::optional;
}    // namespace hpx

#else

#include <hpx/datastructures/detail/optional.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx {

    using hpx::optional_ns::bad_optional_access;
    using hpx::optional_ns::make_optional;
    using hpx::optional_ns::nullopt;
    using hpx::optional_ns::nullopt_t;
    using hpx::optional_ns::optional;
}    // namespace hpx

#include <hpx/config/warnings_suffix.hpp>

#endif
