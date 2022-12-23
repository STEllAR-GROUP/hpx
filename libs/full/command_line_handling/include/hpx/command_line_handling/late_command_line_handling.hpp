//  Copyright (c) 2021-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/runtime_configuration.hpp>

#include <cstddef>

namespace hpx::util {

    HPX_EXPORT int handle_late_commandline_options(
        util::runtime_configuration& ini,
        hpx::program_options::options_description const& options,
        void (*handle_print_bind)(std::size_t),
        void (*handle_list_parcelports)() = nullptr);
}    // namespace hpx::util
