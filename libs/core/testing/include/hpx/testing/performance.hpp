//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/function.hpp>

#include <cstddef>
#include <string>

namespace hpx::util {

    HPX_CORE_EXPORT void perftests_report(std::string const& name,
        std::string const& exec, std::size_t const steps,
        hpx::function<void()>&& test);

    HPX_CORE_EXPORT void perftests_print_times();
}    // namespace hpx::util
