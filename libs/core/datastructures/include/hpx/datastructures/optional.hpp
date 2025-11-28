//  Copyright (c) 2022-2025 Hartmut Kaiser
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

    HPX_CXX_EXPORT using std::bad_optional_access;
    HPX_CXX_EXPORT using std::make_optional;
    HPX_CXX_EXPORT using std::nullopt;
    HPX_CXX_EXPORT using std::nullopt_t;
    HPX_CXX_EXPORT using std::optional;
}    // namespace hpx

#else

#include <hpx/datastructures/detail/optional.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx {

    HPX_CXX_EXPORT using hpx::optional_ns::bad_optional_access;
    HPX_CXX_EXPORT using hpx::optional_ns::make_optional;
    HPX_CXX_EXPORT using hpx::optional_ns::nullopt;
    HPX_CXX_EXPORT using hpx::optional_ns::nullopt_t;
    HPX_CXX_EXPORT using hpx::optional_ns::optional;
}    // namespace hpx

#include <hpx/config/warnings_suffix.hpp>

#endif
