//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/runtime_local.hpp>

#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace hpx {
    struct init_params
    {
        init_params()
        {
            std::strncpy(hpx::local::detail::app_name, HPX_APPLICATION_STRING,
                sizeof(hpx::local::detail::app_name) - 1);
        }

        std::reference_wrapper<hpx::program_options::options_description const>
            desc_cmdline =
                hpx::local::detail::default_desc(HPX_APPLICATION_STRING);
        std::vector<std::string> cfg;
        std::function<void()> startup;
        std::function<void()> shutdown;
    };

    HPX_CORE_EXPORT int init(
        std::function<int(hpx::program_options::variables_map&)> f, int argc,
        char** argv, init_params const& params = init_params());

    HPX_CORE_EXPORT int init(std::function<int(int, char**)> f, int argc,
        char** argv, init_params const& params = init_params());

    HPX_CORE_EXPORT int init(std::function<int()> f, int argc, char** argv,
        init_params const& params = init_params());

    HPX_CORE_EXPORT int init(std::nullptr_t, int argc, char** argv,
        init_params const& params = init_params());

    HPX_CORE_EXPORT int init(init_params const& params = init_params());

    HPX_CORE_EXPORT int finalize(hpx::error_code& ec = throws);
}    // namespace hpx
