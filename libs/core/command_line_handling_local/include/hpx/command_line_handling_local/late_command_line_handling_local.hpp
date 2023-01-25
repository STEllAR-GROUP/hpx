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
#include <string>
#include <vector>

namespace hpx::local::detail {

    HPX_CORE_EXPORT int handle_late_commandline_options(
        util::runtime_configuration& ini,
        hpx::program_options::options_description const& options,
        void (*handle_print_bind)(std::size_t) = nullptr);

    HPX_CORE_EXPORT void set_unknown_commandline_options(
        util::runtime_configuration& ini,
        std::vector<std::string> const& still_unregistered_options);

    HPX_CORE_EXPORT bool handle_full_help(util::runtime_configuration& ini,
        hpx::program_options::options_description const& options);
    HPX_CORE_EXPORT bool handle_late_options(util::runtime_configuration& ini,
        hpx::program_options::variables_map& vm,
        void (*handle_print_bind)(std::size_t) = nullptr);

    HPX_CORE_EXPORT std::string get_full_commandline(
        util::runtime_configuration& ini);
}    // namespace hpx::local::detail
