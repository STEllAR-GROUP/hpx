//    Copyright (c) 2004 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/program_options/config/defines.hpp>
#include <hpx/datastructures/any.hpp>
#include <hpx/datastructures/optional.hpp>

namespace hpx::program_options {

    using any = hpx::any_nonser;
    using hpx::any_cast;

    template <typename T>
    using optional = hpx::optional<T>;
}    // namespace hpx::program_options
