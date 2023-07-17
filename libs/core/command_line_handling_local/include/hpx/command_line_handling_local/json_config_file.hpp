//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/command_line_handling_local/config/defines.hpp>

#if defined(HPX_COMMAND_LINE_HANDLING_HAVE_JSON_CONFIGURATION_FILES)

#include <hpx/command_line_handling_local/parse_command_line_local.hpp>

#include <string>
#include <vector>

namespace hpx::local::detail {

    HPX_CORE_EXPORT std::vector<std::string> read_json_config_file_options(
        std::string const& filename, util::commandline_error_mode error_mode);

}    // namespace hpx::local::detail

#endif
