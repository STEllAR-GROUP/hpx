//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/init_params.hpp>
#include <hpx/modules/init_runtime_local.hpp>
#include <hpx/modules/program_options.hpp>

namespace hpx {

    inline hpx::local::init_params map_params(init_params const& params)
    {
        hpx::local::init_params local_params;
        local_params.desc_cmdline = params.desc_cmdline;
        local_params.cfg = params.cfg;
        local_params.startup = params.startup;
        local_params.shutdown = params.shutdown;
        return local_params;
    }

    int init(std::function<int(hpx::program_options::variables_map&)> f,
        int argc, char** argv, init_params const& params)
    {
        return hpx::local::init(HPX_MOVE(f), argc, argv, map_params(params));
    }

    int init(std::function<int(int, char**)> f, int argc, char** argv,
        init_params const& params)
    {
        return hpx::local::init(HPX_MOVE(f), argc, argv, map_params(params));
    }

    int init(std::function<int()> f, int argc, char** argv,
        init_params const& params)
    {
        return hpx::local::init(HPX_MOVE(f), argc, argv, map_params(params));
    }

    int init(std::nullptr_t, int argc, char** argv, init_params const& params)
    {
        return hpx::local::init(nullptr, argc, argv, map_params(params));
    }

    int init(init_params const& params)
    {
        return hpx::local::init(nullptr, hpx::local::detail::dummy_argc,
            hpx::local::detail::dummy_argv, map_params(params));
    }

    int finalize(hpx::error_code& ec)
    {
        return hpx::local::finalize(ec);
    }

}    // namespace hpx
